"""Hybrid Vector + Knowledge Graph Memory.

Architecture
------------
User text → VectorMemory.add()   (ChromaDB, semantic embeddings)
          → GraphMemory.add()    (NetworkX, rule-based triples)

Query → VectorMemory.retrieve()  (top-K cosine similarity)
      → GraphMemory.retrieve()   (bounded 2-hop graph traversal)
              ↓
        Evidence Fusion
              ↓
        Deduplicate by memory_source_id
              ↓
        Min-Max Normalise each score pool independently
              ↓
        Hybrid Score = α * norm_vector + (1-α) * norm_graph
              ↓
        Rank → Top-K

Score Normalisation
-------------------
Vector and graph scores are on different numerical scales (cosine distance
vs integer keyword-overlap counts).  We normalise each pool independently
using min-max within the candidate set:

    normalised = (s - s_min) / (s_max - s_min)

If all candidates share the same score (s_max == s_min) every normalised
score is set to 1.0.

Missing-source handling
-----------------------
A memory_source_id may appear in only one retriever.  Its score in the
missing retriever is treated as 0.0 (worst possible normalised score),
not as absent.  This is conservative: a source that only one retriever
finds is penalised relative to one that both confirm.

Fusion weight (alpha)
---------------------
    hybrid_score = alpha * norm_vector_score + (1-alpha) * norm_graph_score

alpha=0.5 is the fixed initial value for this baseline.
alpha is NOT tuned against the 50 evaluation cases.
It can be changed via the HYBRID_ALPHA environment variable.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional

from src.memory.base import BaseMemory
from src.memory.vector_memory import VectorMemory
from src.memory.graph_memory import GraphMemory
from src.memory.embeddings import BaseEmbedding, FakeEmbedding
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridMemory(BaseMemory):
    """Combined Vector + Graph memory with score-fusion retrieval.

    Parameters
    ----------
    vector_memory : VectorMemory
        Fully-configured VectorMemory instance.
    graph_memory : GraphMemory
        Fully-configured GraphMemory instance.
    alpha : float
        Fusion weight for vector scores (0 = pure graph, 1 = pure vector).
        Default 0.5. Do NOT tune this against the evaluation dataset.
    top_k : int
        Number of results to return after fusion and ranking.
    """

    def __init__(
        self,
        vector_memory: VectorMemory,
        graph_memory: GraphMemory,
        alpha: float = 0.5,
        top_k: int = 5,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.vector_memory = vector_memory
        self.graph_memory = graph_memory
        self.alpha = alpha
        self.top_k = top_k
        logger.info(
            f"HybridMemory initialised: alpha={alpha}, top_k={top_k}"
        )

    # ── BaseMemory interface ───────────────────────────────────────────────

    def add(self, session_id: str, data: Dict[str, Any]) -> None:
        """Store data in both VectorMemory and GraphMemory."""
        self.vector_memory.add(session_id, data)
        self.graph_memory.add(session_id, data)

    def retrieve(self, query: str, session_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve, fuse, and rank evidence from both memory systems.

        Returns a list of dicts that expose, for every candidate:
            memory_id           – runtime id (from the primary retriever)
            memory_source_id    – stable ground-truth provenance id
            content             – stored text
            vector_score        – raw cosine similarity (None if not retrieved by vector)
            graph_score         – raw graph traversal score (None if not retrieved by graph)
            norm_vector_score   – min-max normalised vector score (0.0 if missing)
            norm_graph_score    – min-max normalised graph score  (0.0 if missing)
            hybrid_score        – α * norm_vector + (1-α) * norm_graph
            retrieval_source    – "vector", "graph", or "both"
            similarity          – alias for hybrid_score (pipeline compatibility)
            metadata            – provenance metadata dict
        """
        # ── 1. Retrieve from each sub-system ──────────────────────────────
        vec_results = self.vector_memory.retrieve(query, top_k=self.top_k * 2)
        graph_results = self.graph_memory.retrieve(query, session_id=session_id)

        # ── 2. Index by memory_source_id ──────────────────────────────────
        # We key on memory_source_id because that is the stable research id.
        # Runtime UUIDs are not used as keys.
        vec_by_source: Dict[str, Dict] = {}
        for r in vec_results:
            sid = r.get("metadata", {}).get("memory_source_id") or r.get("memory_source_id")
            if sid:
                # Keep the highest-scoring entry per source
                if sid not in vec_by_source or r["similarity"] > vec_by_source[sid]["similarity"]:
                    vec_by_source[sid] = r

        graph_by_source: Dict[str, Dict] = {}
        for r in graph_results:
            sid = r.get("memory_source_id") or r.get("metadata", {}).get("memory_source_id")
            if sid:
                if sid not in graph_by_source or r["similarity"] > graph_by_source[sid]["similarity"]:
                    graph_by_source[sid] = r

        # ── 3. Collect all unique source IDs ──────────────────────────────
        all_sources = set(vec_by_source.keys()) | set(graph_by_source.keys())

        if not all_sources:
            return []

        # ── 4. Gather raw scores (missing → None) ─────────────────────────
        raw: Dict[str, Dict] = {}
        for sid in all_sources:
            raw[sid] = {
                "memory_source_id": sid,
                "vec_score": vec_by_source[sid]["similarity"] if sid in vec_by_source else None,
                "graph_score": graph_by_source[sid]["similarity"] if sid in graph_by_source else None,
            }

        # ── 5. Min-Max normalise each pool independently ───────────────────
        def _minmax(values: List[float]) -> List[float]:
            """Normalise a list to [0, 1].  Equal values → 1.0."""
            lo, hi = min(values), max(values)
            if hi == lo:
                return [1.0] * len(values)
            return [(v - lo) / (hi - lo) for v in values]

        vec_raw_vals = [
            raw[sid]["vec_score"]
            for sid in all_sources
            if raw[sid]["vec_score"] is not None
        ]
        graph_raw_vals = [
            raw[sid]["graph_score"]
            for sid in all_sources
            if raw[sid]["graph_score"] is not None
        ]

        # Build lookup: source → normalised score
        vec_sources_ordered = [
            sid for sid in all_sources if raw[sid]["vec_score"] is not None
        ]
        graph_sources_ordered = [
            sid for sid in all_sources if raw[sid]["graph_score"] is not None
        ]

        norm_vec_map: Dict[str, float] = {}
        if vec_raw_vals:
            for sid, ns in zip(vec_sources_ordered, _minmax(vec_raw_vals)):
                norm_vec_map[sid] = ns

        norm_graph_map: Dict[str, float] = {}
        if graph_raw_vals:
            for sid, ns in zip(graph_sources_ordered, _minmax(graph_raw_vals)):
                norm_graph_map[sid] = ns

        # ── 6. Compute hybrid score ────────────────────────────────────────
        # Missing score → 0.0 (conservative: not confirmed by that retriever)
        candidates = []
        for sid in all_sources:
            nv = norm_vec_map.get(sid, 0.0)
            ng = norm_graph_map.get(sid, 0.0)
            hybrid = self.alpha * nv + (1 - self.alpha) * ng

            in_vec = sid in vec_by_source
            in_graph = sid in graph_by_source

            # Select best content + memory_id from whichever retriever found it
            if in_vec and in_graph:
                source_label = "both"
                # Prefer vector content (verbatim user text)
                content = vec_by_source[sid].get("content", "")
                memory_id = vec_by_source[sid].get("memory_id", "")
                metadata = vec_by_source[sid].get("metadata", {})
            elif in_vec:
                source_label = "vector"
                content = vec_by_source[sid].get("content", "")
                memory_id = vec_by_source[sid].get("memory_id", "")
                metadata = vec_by_source[sid].get("metadata", {})
            else:
                source_label = "graph"
                content = graph_by_source[sid].get("content", "")
                memory_id = graph_by_source[sid].get("memory_id", "")
                metadata = graph_by_source[sid].get("metadata", {})

            candidates.append({
                "memory_id": memory_id,
                "memory_source_id": sid,
                "content": content,
                "vector_score": raw[sid]["vec_score"],
                "graph_score": raw[sid]["graph_score"],
                "norm_vector_score": nv,
                "norm_graph_score": ng,
                "hybrid_score": round(hybrid, 6),
                "similarity": round(hybrid, 6),   # pipeline compatibility alias
                "retrieval_source": source_label,
                "metadata": metadata,
            })

        # ── 7. Sort by hybrid score and return top-K ──────────────────────
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return candidates[: self.top_k]

    def update(self, memory_id: str, data: Dict[str, Any]) -> None:
        """Update in both sub-systems by memory_id."""
        self.vector_memory.update(memory_id, data)
        self.graph_memory.update(memory_id, data)

    def forget(self, memory_id: str) -> None:
        """Forget from both sub-systems by memory_id."""
        self.vector_memory.forget(memory_id)
        self.graph_memory.forget(memory_id)

    def reset(self) -> None:
        """Reset both sub-systems completely."""
        self.vector_memory.reset()
        self.graph_memory.reset()
        logger.info("HybridMemory reset: both sub-stores cleared")

    # ── Convenience ───────────────────────────────────────────────────────

    def count(self) -> Dict[str, int]:
        """Return edge/document counts for both sub-stores."""
        return {
            "vector": self.vector_memory.count(),
            "graph": self.graph_memory.count(),
        }
