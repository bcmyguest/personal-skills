"""Slow-learning neocortex: a VAE over text embeddings.

The cortex holds the organisation's *schema* -- the statistical regularities of
what normally gets said and done. It learns slowly, with interleaved training,
and its job in this system is not generation but prediction: how surprising is
this new item given everything the organisation already knows?

Surprise is the reconstruction error (or, optionally, the full negative ELBO,
which is a proper upper bound on -log p(x)). High surprise means the schema
failed -> route to the hippocampus. Low surprise means the schema already
covers it -> nothing worth storing separately.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import CortexConfig, NoveltyConfig


def _mlp(dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@dataclass
class VAEOutput:
    recon: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor


class SlowLearningNeocortex(nn.Module):
    """Beta-VAE over embedding vectors."""

    def __init__(self, config: CortexConfig | None = None) -> None:
        super().__init__()
        self.config = config or CortexConfig()
        c = self.config

        enc_dims = [c.input_dim, *c.hidden_dims]
        self.encoder = _mlp(enc_dims, c.dropout)
        self.to_mu = nn.Linear(enc_dims[-1], c.latent_dim)
        self.to_logvar = nn.Linear(enc_dims[-1], c.latent_dim)

        dec_dims = [c.latent_dim, *reversed(c.hidden_dims)]
        self.decoder = _mlp(dec_dims, c.dropout)
        self.to_recon = nn.Linear(dec_dims[-1], c.input_dim)

    # ------------------------------------------------------------- mechanics

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-10.0, 10.0)

    def decode(self, z: Tensor) -> Tensor:
        return self.to_recon(self.decoder(z))

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: Tensor, *, sample: bool = True) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return VAEOutput(recon=self.decode(z), mu=mu, logvar=logvar, z=z)

    # ------------------------------------------------------------------ loss

    def losses(self, x: Tensor, *, sample: bool = True) -> tuple[Tensor, Tensor]:
        """Per-sample (reconstruction, KL). Both summed over feature dims."""
        out = self.forward(x, sample=sample)
        recon = ((out.recon - x) ** 2).sum(-1)
        kl = -0.5 * (1 + out.logvar - out.mu.pow(2) - out.logvar.exp()).sum(-1)
        return recon, kl

    def elbo_loss(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        recon, kl = self.losses(x)
        loss = (recon + self.config.kl_weight * kl).mean()
        return loss, recon.mean().detach(), kl.mean().detach()

    # ------------------------------------------------------------- inference

    @torch.no_grad()
    def surprise(self, x: Tensor, mode: str = "recon") -> Tensor:
        """Per-sample novelty score. Deterministic (uses mu, not a sample), so
        the same input always yields the same gate decision."""
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        recon, kl = self.losses(x, sample=False)
        if mode == "recon":
            return recon
        if mode == "elbo":
            return recon + self.config.kl_weight * kl
        raise ValueError(f"unknown novelty mode: {mode!r}")

    @torch.no_grad()
    def latent(self, x: Tensor) -> Tensor:
        """Deterministic latent code (the posterior mean)."""
        self.eval()
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        mu, _ = self.encode(x)
        return mu.squeeze(0) if squeeze else mu

    # -------------------------------------------------------------- training

    def fit(
        self,
        x: Tensor,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        verbose: bool = False,
    ) -> list[float]:
        """Slow learning: many small gradient steps over the whole corpus.

        Deliberately plain -- swap in a real training loop (schedulers, early
        stopping, validation split) for anything beyond a proof of concept.
        """
        c = self.config
        epochs = c.epochs if epochs is None else epochs
        batch_size = c.batch_size if batch_size is None else batch_size
        opt = optimizer or torch.optim.Adam(self.parameters(), lr=lr or c.learning_rate)

        self.train()
        history: list[float] = []
        n = x.shape[0]
        for epoch in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch_size):
                batch = x[perm[i : i + batch_size]]
                loss, _, _ = self.elbo_loss(batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * batch.shape[0]
            history.append(total / max(n, 1))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch + 1:>4}/{epochs}  loss={history[-1]:.4f}")
        self.eval()
        return history


class NoveltyGate:
    """Turns a surprise score into a store/prune decision.

    Two modes:
      * fixed threshold -- simple, but the right value drifts as the cortex
        trains and depends on the embedding model's scale;
      * self-calibrating -- 'novel' means 'in the top (1-q) of recent surprise'.
        Scale-free and adapts as the schema improves, which is what you want
        for a system that keeps learning.
    """

    def __init__(self, config: NoveltyConfig | None = None) -> None:
        self.config = config or NoveltyConfig()
        self._window: deque[float] = deque(maxlen=self.config.window)

    @property
    def observed(self) -> int:
        return len(self._window)

    @property
    def threshold(self) -> float:
        c = self.config
        if not c.calibrate:
            if c.threshold is None:
                raise ValueError("threshold must be set when calibrate=False")
            return c.threshold
        if self.observed < c.warmup:
            return float("-inf")  # cold start: everything is novel
        q = torch.tensor(sorted(self._window)).quantile(c.quantile)
        return float(q)

    def observe(self, score: float) -> None:
        """Record a score without making a decision (e.g. during calibration)."""
        self._window.append(float(score))

    def __call__(self, score: float, *, observe: bool = True) -> tuple[bool, float]:
        """Returns (is_novel, threshold_used). Threshold is computed *before*
        the new score joins the window, so an item cannot move its own bar."""
        threshold = self.threshold
        if observe:
            self.observe(score)
        return float(score) > threshold, threshold

    def prime(self, scores: Tensor) -> None:
        """Seed the window from a batch (e.g. the initial training corpus)."""
        for s in scores.reshape(-1).tolist():
            self.observe(s)
