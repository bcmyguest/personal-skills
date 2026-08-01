"""Text -> vector. Pluggable, with a dependency-free default.

The memory system only needs an object with `dim` and `encode(texts) -> Tensor`.
Swap in whatever the organisation already uses (sentence-transformers, an API
embedding model, a fine-tuned encoder); nothing downstream changes.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> Tensor: ...


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Deterministic hashed bag-of-ngrams + fixed random projection.

    Not a semantic model -- it captures lexical overlap only. It exists so the
    demo and the tests run offline, reproducibly, with no model download. For
    real use, replace it: the novelty signal is only as good as the geometry of
    the embedding space it is measured in.
    """

    def __init__(
        self,
        dim: int = 384,
        *,
        buckets: int = 4096,
        ngram: int = 2,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.buckets = buckets
        self.ngram = ngram
        generator = torch.Generator().manual_seed(seed)
        # Fixed projection: shared across calls so encodings stay comparable.
        self.projection = torch.randn(buckets, dim, generator=generator) / (dim**0.5)

    def _features(self, text: str) -> Tensor:
        tokens = _tokenize(text)
        grams = list(tokens)
        for n in range(2, self.ngram + 1):
            grams += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        vec = torch.zeros(self.buckets)
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "little") % self.buckets
            vec[idx] += 1.0
        return vec

    def encode(self, texts: Sequence[str]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]
        feats = torch.stack([self._features(t) for t in texts])
        feats = torch.log1p(feats)  # damp repeated-token dominance
        out = feats @ self.projection
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class SentenceTransformerEmbedder:
    """Thin adapter over sentence-transformers, if it is installed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is not installed; "
                "use HashingEmbedder or `pip install sentence-transformers`"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> Tensor:  # pragma: no cover - optional
        if isinstance(texts, str):
            texts = [texts]
        return torch.as_tensor(
            self._model.encode(list(texts), normalize_embeddings=True),
            dtype=torch.float32,
        )
