"""Systems consolidation: hippocampus -> neocortex transfer.

This is the half of CLS that people usually skip, and the half that makes the
architecture worth building. Three things happen in a consolidation pass:

1. **Replay.** Sample the hippocampal energy landscape with Langevin dynamics
   using the exact score (see energy.py). Samples are drawn from the *smoothed*
   memory distribution, so the cortex learns the neighbourhood around each
   episode rather than memorising the episode itself.

2. **Interleaved training.** Train the cortex on replay mixed with new data.
   Interleaving is the mechanism that prevents catastrophic forgetting -- it is
   the entire reason biological memory bothers with a two-speed design, and
   training on new data alone would defeat the purpose of the architecture.

3. **Pruning of the now-predictable.** Re-score every stored memory through the
   updated cortex. Episodic traces the schema can now reconstruct have been
   absorbed into semantic knowledge and are dropped from the fast store.
   Evergreen records are exempt by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch import Tensor

from .config import ConsolidationConfig, HopfieldConfig
from .energy import langevin_replay
from .hippocampus import ModernHopfieldNetwork
from .neocortex import NoveltyGate, SlowLearningNeocortex
from .records import utcnow
from .store import MemoryStore, SweepReport


@dataclass
class ConsolidationReport:
    replayed: int
    new_items: int
    epochs: int
    loss_before: float
    loss_after: float
    pruned_predicted: int
    pruned_ids: list[str] = field(default_factory=list)
    sweep: SweepReport | None = None

    @property
    def improved(self) -> bool:
        return self.loss_after < self.loss_before


class ConsolidationEngine:
    """Sleep, essentially."""

    def __init__(
        self,
        cortex: SlowLearningNeocortex,
        store: MemoryStore,
        *,
        gate: NoveltyGate | None = None,
        config: ConsolidationConfig | None = None,
    ) -> None:
        self.cortex = cortex
        self.store = store
        self.gate = gate
        self.config = config or ConsolidationConfig()

    # ---------------------------------------------------------------- replay

    def _latent_landscape(self) -> ModernHopfieldNetwork:
        """A Hopfield view of the memory set in *latent* space.

        Replay has to happen in a space the cortex can read, and the cortex
        reads latents -- but the hippocampus proper is keyed on
        pattern-separated codes (see pattern_separation), which the decoder has
        never seen. So the energy landscape is rebuilt over the stored latents,
        carrying the same salience priors, and sampled there.

        Patterns are deliberately not normalised here: normalising would move
        them off the manifold the decoder was trained on. The Gibbs measure is
        then still a Gaussian mixture centred on the latents, with mixing
        weights tilted by exp(beta * ||z_i||^2 / 2) -- so Langevin sampling
        remains principled, it just favours higher-norm latents slightly. See
        the derivation in energy.py.
        """
        landscape = ModernHopfieldNetwork(
            self.store.latents().shape[-1],
            HopfieldConfig(
                beta=self.store.mhn.config.beta, normalize_patterns=False
            ),
        )
        landscape.write(self.store.latents(), self.store.mhn.log_prior.clone())
        return landscape

    def replay(self, n: int | None = None, *, generator: torch.Generator | None = None) -> Tensor:
        """Generate embedding-space replay samples from the hippocampus.

        Latents are sampled from the memory distribution, then pushed through
        the VAE decoder: a hippocampal trace reinstating a cortical pattern,
        which is the literal mechanism the CLS account proposes.
        """
        c = self.config
        n = c.replay_batch if n is None else n
        if len(self.store) == 0 or n <= 0:
            return torch.empty(0, self.cortex.config.input_dim)

        self.store.refresh_priors()
        latents = langevin_replay(
            self._latent_landscape(),
            n,
            steps=c.replay_steps,
            step_size=c.replay_step_size,
            generator=generator,
        )
        with torch.no_grad():
            return self.cortex.decode(latents)

    # ----------------------------------------------------------- consolidate

    def consolidate(
        self,
        new_data: Tensor | None = None,
        *,
        epochs: int = 5,
        lr: float | None = None,
        now: datetime | None = None,
        generator: torch.Generator | None = None,
    ) -> ConsolidationReport:
        """One consolidation cycle: replay + interleaved training + pruning."""
        now = now or utcnow()
        c = self.config

        new_data = (
            torch.empty(0, self.cortex.config.input_dim) if new_data is None else new_data
        )
        n_new = int(new_data.shape[0])

        # Size replay so it makes up `interleave_ratio` of the batch.
        if n_new > 0 and c.interleave_ratio < 1.0:
            n_replay = int(round(n_new * c.interleave_ratio / max(1e-6, 1 - c.interleave_ratio)))
        else:
            n_replay = c.replay_batch
        replayed = self.replay(max(n_replay, 0), generator=generator)

        batch = torch.cat([t for t in (new_data, replayed) if t.numel() > 0], dim=0)
        if batch.shape[0] == 0:
            return ConsolidationReport(0, 0, 0, 0.0, 0.0, 0)

        with torch.no_grad():
            loss_before = float(self.cortex.elbo_loss(batch)[0])
        self.cortex.fit(batch, epochs=epochs, lr=lr)
        with torch.no_grad():
            loss_after = float(self.cortex.elbo_loss(batch)[0])

        pruned = self.prune_predicted(now=now) if c.prune_predicted else []
        sweep = self.store.sweep(now)

        return ConsolidationReport(
            replayed=int(replayed.shape[0]),
            new_items=n_new,
            epochs=epochs,
            loss_before=loss_before,
            loss_after=loss_after,
            pruned_predicted=len(pruned),
            pruned_ids=pruned,
            sweep=sweep,
        )

    # ----------------------------------------------------------------- prune

    def prune_predicted(
        self, *, threshold: float | None = None, now: datetime | None = None
    ) -> list[str]:
        """Drop episodic memories the cortex now reconstructs below threshold.

        This is the closing of the loop: the hippocampus is a staging area, and
        anything the schema has absorbed should not keep occupying it. Without
        this step the fast store grows without bound and its attractors blur
        together (see ModernHopfieldNetwork.separation).
        """
        if len(self.store) == 0:
            return []

        if threshold is None:
            if self.gate is None:
                raise ValueError("pass a threshold or construct with a NoveltyGate")
            threshold = self.gate.threshold
        if threshold == float("-inf"):
            return []  # gate still in warmup; nothing is reliably predictable yet

        mode = self.gate.config.mode if self.gate else "recon"
        embeddings = self.store.embeddings()
        scores = self.cortex.surprise(embeddings, mode)

        doomed = [
            r.id
            for r, s in zip(self.store.records, scores.tolist())
            if s <= threshold and not (r.is_evergreen and self.config.protect_evergreen)
        ]
        removed = self.store.remove(doomed)
        self.store.refresh_priors(now)
        return removed

    # ------------------------------------------------------------ diagnostics

    def predictability(self) -> dict:
        """Per-record surprise under the current cortex -- shows which memories
        are on their way to being consolidated away."""
        if len(self.store) == 0:
            return {}
        mode = self.gate.config.mode if self.gate else "recon"
        scores = self.cortex.surprise(self.store.embeddings(), mode)
        return {r.id: float(s) for r, s in zip(self.store.records, scores.tolist())}
