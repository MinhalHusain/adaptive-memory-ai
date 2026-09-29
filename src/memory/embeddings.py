"""Embedding provider abstractions.

Provides a configurable embedding interface so the vector store is not
coupled to a specific model or API.  Three concrete implementations:

* FakeEmbedding   – deterministic, no dependencies, for unit tests.
* SentenceTransformerEmbedding – local, free (requires sentence-transformers).
* OpenAIEmbedding – API-based (requires openai + API key).
"""

from abc import ABC, abstractmethod
from typing import List
import hashlib
import math
import os


class BaseEmbedding(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts and return a list of float vectors."""
        ...

    def embed_single(self, text: str) -> List[float]:
        """Convenience wrapper for embedding a single text."""
        return self.embed([text])[0]


# ── Deterministic fake for tests ──────────────────────────────────────────


class FakeEmbedding(BaseEmbedding):
    """Deterministic embedding using character-trigram hashing.

    Produces vectors where texts sharing more character trigrams are
    closer together.  Useful for unit tests that need repeatable
    similarity rankings without any external model.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        t = text.lower()
        for i in range(max(1, len(t) - 2)):
            trigram = t[i : i + 3]
            h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
            idx = h % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# ── Sentence-Transformers (local, free) ───────────────────────────────────


class SentenceTransformerEmbedding(BaseEmbedding):
    """Local embedding via the sentence-transformers library.

    Default model: ``all-MiniLM-L6-v2`` (384-d, fast, good quality).
    Override via the ``EMBEDDING_MODEL`` environment variable.

    Requires: ``pip install sentence-transformers``
    """

    def __init__(self, model_name: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed.  "
                "Run: pip install sentence-transformers"
            )
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(self.model_name)
        self._dim: int = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


# ── OpenAI Embeddings (API-based) ─────────────────────────────────────────


class OpenAIEmbedding(BaseEmbedding):
    """Embedding via the OpenAI API.

    Default model: ``text-embedding-3-small`` (1 536-d).
    Override via ``EMBEDDING_MODEL`` environment variable.

    Requires: ``pip install openai`` + ``OPENAI_API_KEY`` set.
    """

    _DIMENSION_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is not installed.  Run: pip install openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY."
            )
        self.model = model or os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.client = OpenAI(api_key=self.api_key)
        self._dim = self._DIMENSION_MAP.get(self.model, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
