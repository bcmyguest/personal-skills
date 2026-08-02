"""Text -> vector. Pluggable, with a dependency-free default.

The memory system only needs an object with `dim` and `encode(texts) -> Tensor`.
Swap in whatever the organisation already uses (sentence-transformers, an API
embedding model, a fine-tuned encoder); nothing downstream changes.
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
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


class LatentSemanticEmbedder:
    """TF-IDF over character and word n-grams, reduced by truncated SVD (LSA).

    The middle option between `HashingEmbedder` (lexical only, no fitting) and a
    real sentence encoder (needs a model download). It is *fitted on your
    corpus*, so it learns which terms actually discriminate in your domain, and
    the SVD step means two documents sharing no exact token can still land close
    if they co-occur with the same vocabulary.

    Use this when a sentence encoder is unavailable. It is a large improvement
    on hashing -- measured on LoCoMo against ground-truth evidence turns,
    recall@5 goes 0.045 -> 0.174 and recall@1 goes 0.024 -> 0.121, roughly 4x --
    but it is still a bag of n-grams with no word order and no compositional
    semantics, and 0.174 is not a good number in absolute terms. If you can
    reach a sentence encoder, use that instead; this is the floor, not the goal.

    Fit before use:

        embedder = LatentSemanticEmbedder(dim=256)
        embedder.fit(corpus_texts)
    """

    def __init__(
        self,
        dim: int = 256,
        *,
        max_features: int = 20_000,
        min_df: int = 2,
        char_ngrams: tuple[int, int] | None = (3, 5),
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.max_features = max_features
        self.min_df = min_df
        self.char_ngrams = char_ngrams
        self.seed = seed
        self._vocab: dict[str, int] = {}
        self._idf: Tensor | None = None
        self._components: Tensor | None = None
        self.rank: int = 0
        # Components the corpus actually supports. Below `dim` means the
        # trailing coordinates are structurally zero; fit() warns when so.

    # ---------------------------------------------------------------- features

    def _terms(self, text: str) -> list[str]:
        tokens = _tokenize(text)
        terms = list(tokens)
        terms += [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
        if self.char_ngrams:
            lo, hi = self.char_ngrams
            padded = f" {text.lower().strip()} "
            for n in range(lo, hi + 1):
                terms += [padded[i : i + n] for i in range(len(padded) - n + 1)]
        return terms

    def _counts(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for term in self._terms(text):
            idx = self._vocab.get(term)
            if idx is not None:
                counts[idx] = counts.get(idx, 0.0) + 1.0
        return counts

    def _tfidf(self, texts: Sequence[str], *, sparse: bool = False) -> Tensor:
        """Row-normalised TF-IDF.

        `sparse=True` returns a COO tensor. The dense path allocates
        len(texts) x len(vocab) floats, which is fine for an encode batch but
        not for a fit: measured, a 10k-document corpus with a 20k vocabulary
        peaked at 2.1 GB, extrapolating to ~8 GB in one allocation at 100k.
        """
        if self._idf is None:
            raise RuntimeError("call fit() before encoding")
        if not sparse:
            out = torch.zeros(len(texts), len(self._vocab))
            for row, text in enumerate(texts):
                for idx, count in self._counts(text).items():
                    out[row, idx] = 1.0 + math.log(count)
            out *= self._idf
            return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for row, text in enumerate(texts):
            counts = self._counts(text)
            if not counts:
                continue
            weights = {
                idx: (1.0 + math.log(c)) * float(self._idf[idx])
                for idx, c in counts.items()
            }
            norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
            for idx, value in weights.items():
                rows.append(row)
                cols.append(idx)
                vals.append(value / norm)
        indices = torch.tensor([rows, cols], dtype=torch.long)
        return torch.sparse_coo_tensor(
            indices, torch.tensor(vals), (len(texts), len(self._vocab))
        ).coalesce()

    # -------------------------------------------------------------------- fit

    def fit(self, texts: Sequence[str]) -> "LatentSemanticEmbedder":
        """Build the vocabulary and the SVD basis from a corpus."""
        texts = list(texts)
        if not texts:
            raise ValueError("cannot fit on an empty corpus")

        document_frequency: dict[str, int] = {}
        for text in texts:
            for term in set(self._terms(text)):
                document_frequency[term] = document_frequency.get(term, 0) + 1

        keep = [t for t, df in document_frequency.items() if df >= self.min_df]
        keep.sort(key=lambda t: (-document_frequency[t], t))
        keep = keep[: self.max_features]
        self._vocab = {term: i for i, term in enumerate(keep)}
        if not self._vocab:
            raise ValueError("no terms survived min_df; lower it or add data")

        n = len(texts)
        self._idf = torch.tensor(
            [math.log((1 + n) / (1 + document_frequency[t])) + 1.0 for t in keep]
        )

        matrix = self._tfidf(texts, sparse=True)
        # Randomised range finder, then an exact SVD on the small projection.
        # Cheaper than a full SVD on a 20k-column matrix and accurate enough at
        # the ranks used here.
        k = min(self.dim, matrix.shape[0], matrix.shape[1])
        generator = torch.Generator().manual_seed(self.seed)
        # Oversample by 32 (was 16) and re-orthonormalise after every half
        # step. Without the intermediate QR, two power iterations in float32
        # collapse the trailing directions toward the dominant singular vector:
        # measured against an exact dense SVD, per-component |cos| ran 1.00 for
        # the leading eight but as low as 0.07 in the tail, and at the default
        # input_dim=1024 that tail is most of the embedding.
        width = min(k + 32, matrix.shape[1])
        omega = torch.randn(matrix.shape[1], width, generator=generator)
        sample, _ = torch.linalg.qr(torch.sparse.mm(matrix, omega))
        transposed = matrix.t().coalesce()
        for _ in range(2):
            projected, _ = torch.linalg.qr(torch.sparse.mm(transposed, sample))
            sample, _ = torch.linalg.qr(torch.sparse.mm(matrix, projected))
        q = sample
        # (q^T M) = (M^T q)^T, and only the sparse-dense product is supported.
        _, _, vh = torch.linalg.svd(
            torch.sparse.mm(transposed, q).T, full_matrices=False
        )
        components = vh[:k].T.contiguous()

        # Honour the requested dim even when the corpus supports fewer
        # components: pad with zero columns rather than silently returning
        # narrower vectors. A corpus-dependent embedding width is a nasty
        # footgun -- it would make the cortex's input_dim depend on how much
        # text happened to be available at bootstrap.
        if components.shape[1] < self.dim:
            padding = torch.zeros(components.shape[0], self.dim - components.shape[1])
            components = torch.cat([components, padding], dim=1)
        self.rank = int(k)
        self._components = components
        if self.rank < self.dim:
            warnings.warn(
                f"corpus of {len(texts)} documents supports only {self.rank} of "
                f"{self.dim} components; the rest are structurally zero",
                stacklevel=2,
            )
        return self

    @property
    def is_fitted(self) -> bool:
        return self._components is not None

    def encode(self, texts: Sequence[str]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]
        if self._components is None:
            raise RuntimeError("call fit() before encoding")
        out = self._tfidf(list(texts)) @ self._components
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
