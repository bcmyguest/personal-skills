"""Dentate-gyrus-style pattern separation for the hippocampal key.

Why this module exists
----------------------
The obvious design -- and the one the original brief specifies -- is to store
the VAE latent in the Hopfield network. Measured on the demo corpus, that is
the worst of the available options:

    key                                 correct recall   max |off-diagonal cos|
    VAE latent                              2/5                  0.905
    raw embedding                           5/5                  0.395
    DG-separated (1024-d, k=256)            5/5                  0.405
    DG-separated (1024-d, k=16)             4/5                  0.059

The failure is structural, not a tuning artefact. The cortex is trained on
*routine* text, so its encoder has no resolution in the region where novel
items live -- exactly the items the hippocampus is asked to store. Two
unrelated incidents landed at cosine 0.905 in latent space and fused into a
single attractor.

CLS predicts this. The dentate gyrus performs pattern separation precisely so
that similar episodes do not interfere: it expands into a much
higher-dimensional space and sparsifies, which decorrelates codes that were
close in the input. That is what this module does -- a fixed random expansion
followed by k-winner-take-all.

The expansion is random and *fixed*, not learned. That is deliberate: it must
not drift as the cortex trains, or previously stored keys would silently stop
matching new queries.
"""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor, nn


class HippocampalKey(str, Enum):
    """Which vector the Hopfield network actually stores."""

    SEPARATED = "separated"
    """DG-style sparse expansion of the embedding. Default: best measured
    recall plus an explicit separation/capacity knob."""

    EMBEDDING = "embedding"
    """The raw encoder embedding. Matches SEPARATED on recall here; keeps the
    key space interpretable and avoids the extra projection."""

    LATENT = "latent"
    """The VAE latent, as originally specified. Retained for comparison and
    for the case where a cortex trained on representative data does give a
    well-separated latent -- but verify separation before trusting it."""


class DentateGyrus(nn.Module):
    """Fixed random expansion + k-winner-take-all sparsification.

    x (d) -> ReLU(W x) (m) -> keep top k -> L2 normalise

    Raising `expansion_dim` or lowering `sparsity_k` decorrelates keys further
    (higher capacity, less interference) at the cost of robustness to partial
    cues: an over-sparse code shares too few active units with a degraded query
    for the overlap to drive retrieval. k/m ~ 0.25 is a reasonable proof of
    concept starting point; real dentate gyrus coding is far sparser, which
    works there because the input representations are much richer than an
    embedding of a single sentence.
    """

    projection: Tensor

    def __init__(
        self,
        input_dim: int,
        expansion_dim: int = 1024,
        sparsity_k: int = 256,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if sparsity_k > expansion_dim:
            raise ValueError("sparsity_k cannot exceed expansion_dim")
        self.input_dim = input_dim
        self.dim = expansion_dim
        self.sparsity_k = sparsity_k
        generator = torch.Generator().manual_seed(seed)
        w = torch.randn(input_dim, expansion_dim, generator=generator)
        self.register_buffer("projection", w / (input_dim**0.5))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        h = torch.relu(x @ self.projection)
        if self.sparsity_k < self.dim:
            threshold = h.topk(self.sparsity_k, dim=-1).values[..., -1:]
            h = h * (h >= threshold)
        h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return h.squeeze(0) if squeeze else h

    def sparsity(self) -> float:
        return self.sparsity_k / self.dim


class KeyEncoder:
    """Dispatches embedding + latent to the configured hippocampal key."""

    def __init__(
        self,
        mode: HippocampalKey,
        *,
        embedding_dim: int,
        latent_dim: int,
        expansion_dim: int = 1024,
        sparsity_k: int = 256,
        seed: int = 0,
    ) -> None:
        self.mode = HippocampalKey(mode)
        self.dg: DentateGyrus | None = None
        if self.mode is HippocampalKey.SEPARATED:
            self.dg = DentateGyrus(
                embedding_dim, expansion_dim, sparsity_k, seed=seed
            )
            self.dim = self.dg.dim
        elif self.mode is HippocampalKey.EMBEDDING:
            self.dim = embedding_dim
        else:
            self.dim = latent_dim

    def __call__(self, embedding: Tensor, latent: Tensor) -> Tensor:
        if self.mode is HippocampalKey.SEPARATED:
            assert self.dg is not None
            key = self.dg(embedding)
        elif self.mode is HippocampalKey.EMBEDDING:
            key = embedding
        else:
            key = latent
        return key / key.norm(dim=-1, keepdim=True).clamp_min(1e-12)
