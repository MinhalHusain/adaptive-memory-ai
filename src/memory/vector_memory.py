"""Vector-based persistent memory using ChromaDB.

Architecture
------------
User text --> Embedding model --> vector --> ChromaDB (persistent on disk)
Query text --> Embedding model --> vector --> Top-K cosine similarity search
                                             --> List of {memory_id, content,
                                                          similarity, metadata}

Memory extraction policy (baseline)
-----------------------------------
Every user message is stored as a single memory unit.  No importance
scoring, compression, or deduplication is applied.  This keeps the
baseline simple and measurable; more sophisticated policies belong
to the future Adaptive Memory implementation.

Storage
-------
ChromaDB PersistentClient writes to ``persist_dir`` on disk.  Data
survives Python process restarts.

Why ChromaDB?
  - Local, zero-config, no external server.
  - Persistent to disk out of the box.
  - Native cosine / L2 similarity search.
  - Metadata filtering support.
  - Ideal for single-machine research prototyping.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb

from src.memory.base import BaseMemory
from src.memory.embeddings import BaseEmbedding
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorMemory(BaseMemory):
    """Persistent vector memory backed by ChromaDB.

    Parameters
    ----------
    embedding : BaseEmbedding
        The embedding provider to use for encoding text.
    persist_dir : str
        Directory on disk where ChromaDB stores data.
    collection_name : str
        Name of the ChromaDB collection.
    top_k : int
        Default number of results to return from retrieval.
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        persist_dir: str = "./data/vector_store",
        collection_name: str = "memories",
        top_k: int = 5,
    ):
        self.embedding = embedding
        self.persist_dir = persist_dir
        self.top_k = top_k
        self._collection_name = collection_name

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorMemory initialized: collection='{collection_name}', "
            f"persist_dir='{persist_dir}', top_k={top_k}, "
            f"existing_memories={self.collection.count()}"
        )

    # ── BaseMemory interface ──────────────────────────────────────────────

    def add(self, session_id: str, data: Dict[str, Any]) -> None:
        """Store a user message as a memory unit.

        Baseline policy: every non-empty ``user_input`` is stored
        verbatim with its session id and timestamp.
        """
        content = data.get("user_input", "")
        if not content or not content.strip():
            return

        memory_id = str(uuid.uuid4())
        emb = self.embedding.embed_single(content)

        metadata: Dict[str, Any] = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "conversation",
        }
        
        # Merge any extra metadata (like memory_source_id)
        extra_metadata = data.get("metadata", {})
        for k, v in extra_metadata.items():
            if isinstance(v, (str, int, float, bool)):
                metadata[k] = v

        self.collection.add(
            ids=[memory_id],
            embeddings=[emb],
            documents=[content],
            metadatas=[metadata],
        )
        logger.debug(f"Stored memory {memory_id}: {content[:80]}")

    def retrieve(
        self, query: str, top_k: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Return the top-K most similar memories to *query*.

        Each returned dict contains:
        - memory_id   – unique id
        - content     – stored text
        - similarity  – cosine similarity (1 = identical)
        - metadata    – session_id, timestamp, etc.
        """
        k = top_k or self.top_k
        count = self.collection.count()
        if count == 0:
            return []
        k = min(k, count)

        query_emb = self.embedding.embed_single(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
        )

        memories: List[Dict[str, Any]] = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                dist = (
                    results["distances"][0][i]
                    if results.get("distances")
                    else None
                )
                similarity = round(1.0 - dist, 6) if dist is not None else None
                memories.append(
                    {
                        "memory_id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "similarity": similarity,
                        "metadata": (
                            results["metadatas"][0][i]
                            if results.get("metadatas")
                            else {}
                        ),
                    }
                )
        return memories

    def update(self, memory_id: str, data: Dict[str, Any]) -> None:
        """Update the content (and re-embed) of an existing memory."""
        content = data.get("content", "")
        if not content or not content.strip():
            return

        emb = self.embedding.embed_single(content)
        metadata = data.get("metadata", {})
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        self.collection.update(
            ids=[memory_id],
            embeddings=[emb],
            documents=[content],
            metadatas=[metadata],
        )
        logger.debug(f"Updated memory {memory_id}")

    def forget(self, memory_id: str) -> None:
        """Remove a specific memory from the store."""
        self.collection.delete(ids=[memory_id])
        logger.debug(f"Deleted memory {memory_id}")

    def reset(self) -> None:
        """Drop and recreate the collection, clearing all memories."""
        self.client.delete_collection(self._collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorMemory reset: all memories cleared")

    # ── Helpers ───────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of memories currently stored."""
        return self.collection.count()
