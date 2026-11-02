"""Pattern completion: settle a partial cue onto an attractor state.

Two cue types are supported, and the difference matters:

  * **Corrupted cue** (`recall`): a text fragment is embedded and encoded. The
    resulting latent is a noisy version of the stored pattern, and every
    coordinate is allowed to move during settling. This is the normal path for
    natural-language queries.

  * **Masked cue** (`complete`): some latent coordinates are known exactly and
    the rest are unknown. Known coordinates are clamped at every iteration and
    only the unknown ones settle. This is pattern completion in the strict
    associative-memory sense, and it is the right path for structured partial
    records ("incident in region eu-west, cause unknown").

Retrieval temperature doubles as a level-of-abstraction control: high beta
returns the single closest episode, low beta returns a metastable mixture that
averages over a family of related memories -- the organisational "gist".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import torch
from torch import Tensor

from .embeddings import Embedder
from .energy import BasinReport, basin_depth
from .hippocampus import RetrievalTrace
from .neocortex import SlowLearningNeocortex
from .pattern_separation import KeyEncoder
from .records import MemoryRecord, utcnow
from .store import MemoryStore


@dataclass
class Recollection:
    """One retrieved memory with its association strength."""

    record: MemoryRecord
    weight: float
    """Softmax weight at the settled state -- the share of the attractor this
    memory accounts for."""
    similarity: float

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        head = self.record.text
        head = head if len(head) <= 60 else head[:57] + "..."
        return f"<{self.weight:.3f} {self.record.persistence.value[:4]} {head!r}>"


@dataclass
class RecallResult:
    query: str | None
    results: list[Recollection]
    trace: RetrievalTrace
    basin: BasinReport
    beta: float

    @property
    def top(self) -> Recollection | None:
        return self.results[0] if self.results else None

    @property
    def is_gist(self) -> bool:
        """The state settled into a mixture over several memories rather than a
        single episode -- a schema-level answer, not a specific recollection."""
        return self.trace.is_mixture

    @property
    def confidence(self) -> float:
        """Share of the attractor held by the top memory, discounted when the
        cue sits outside every basin (likely confabulation)."""
        if not self.results:
            return 0.0
        penalty = 0.5 if self.basin.is_confabulation else 1.0
        return self.results[0].weight * penalty


class PatternCompleter:
    """Read path: cue -> settled attractor -> ranked memories."""

    def __init__(
        self,
        embedder: Embedder,
        cortex: SlowLearningNeocortex,
        store: MemoryStore,
        key_encoder: KeyEncoder,
    ) -> None:
        self.embedder = embedder
        self.cortex = cortex
        self.store = store
        self.key_encoder = key_encoder

    # ------------------------------------------------------------------ cues

    def encode_cue(self, text: str) -> Tensor:
        """Text -> hippocampal key, using the same encoder the write path used."""
        embedding = self.embedder.encode([text])[0]
        latent = self.cortex.latent(embedding)
        return self.key_encoder(embedding, latent)

    # -------------------------------------------------------------- settling

    def _settle(
        self,
        cue: Tensor,
        *,
        mask: Tensor | None,
        beta: float | None,
        top_k: int,
        query: str | None,
        reinforce: bool,
        now: datetime | None,
    ) -> RecallResult:
        if len(self.store) == 0:
            raise RuntimeError("memory is empty -- ingest something first")

        now = now or utcnow()
        # Time-accurate weighting: decay is applied lazily, at read time.
        self.store.refresh_priors(now)

        mhn = self.store.mhn
        trace = mhn.retrieve(cue, mask=mask, beta=beta)

        k = min(top_k, len(self.store))
        # Ranked on the LOGITS at the settled state, not on `trace.weights`.
        # The two are the same ordering in exact arithmetic -- the weights are
        # softmax(logits) -- but not in float32. Once beta * (cosine gap)
        # exceeds ~88 the softmax flushes to exactly 0.0, and at the shipped
        # default of beta=128 that is nearly the whole store: 38 of 40 patterns
        # on a 64-d synthetic, and every question on LoCoMo. `topk` then breaks
        # the resulting tie by storage order, so every rank below the first was
        # returned in *insertion order* rather than by similarity. Measured on
        # LoCoMo conversation 1 at beta=128, hit@10 was 0.327 ranked by weights
        # against 0.418 for the identical store ranked by logits.
        #
        # Ranking on logits is exact at every beta; widening the softmax to
        # float64 would only move the cliff from beta*gap ~88 to ~700.
        idx = torch.topk(mhn.logits(trace.state, beta), k).indices
        weights = trace.weights[idx]
        sims = trace.state @ mhn.patterns.T

        results = [
            Recollection(
                record=self.store.record_at(int(i)),
                weight=float(w),
                similarity=float(sims[int(i)]),
            )
            for w, i in zip(weights, idx)
        ]

        # Basin is measured BEFORE reinforcement, so the trace and the basin
        # describe the same prior vector.
        #
        # Which point to measure at depends on the cue type:
        #   free cue   -> the CUE. The settled state is inside a basin by
        #                 construction (it is a softmax average of stored
        #                 patterns), so measuring there reports depth ~0 for
        #                 every query, including nonsense.
        #   masked cue -> the SETTLED STATE. A deliberately occluded cue is far
        #                 from every memory by construction, so measuring there
        #                 flags perfect completions as confabulations. The
        #                 settled state is still a real test here because the
        #                 known coordinates stay clamped, so it cannot simply
        #                 fall into the nearest basin.
        # Either way the point is normalised: a masked cue has norm
        # ~sqrt(keep_fraction), which would otherwise depress top_similarity.
        basin_point = trace.state if mask is not None else cue
        basin = basin_depth(
            mhn, basin_point / basin_point.norm().clamp_min(1e-12), beta
        )

        if reinforce and results:
            # Reconsolidation: recall makes the winning memory more durable.
            results[0].record.reinforce(
                now,
                gain=self.store.decay.reinforcement_gain,
                max_strength=self.store.decay.max_strength,
                reset_clock=self.store.decay.reinforce_on_recall,
            )
            self.store.refresh_priors(now)

        return RecallResult(
            query=query,
            results=results,
            trace=trace,
            # Measured at the cue, not the settled state: after settling the
            # state is inside a basin by construction, which would report
            # depth 0 for every query including nonsense ones.
            basin=basin,
            beta=mhn._beta(beta),
        )

    # ----------------------------------------------------------------- public

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        beta: float | None = None,
        reinforce: bool = True,
        now: datetime | None = None,
    ) -> RecallResult:
        """Free recall from a natural-language cue (the corrupted-cue path)."""
        return self._settle(
            self.encode_cue(query),
            mask=None,
            beta=beta,
            top_k=top_k,
            query=query,
            reinforce=reinforce,
            now=now,
        )

    def complete(
        self,
        partial: Tensor,
        mask: Tensor,
        *,
        top_k: int = 5,
        beta: float | None = None,
        reinforce: bool = False,
        now: datetime | None = None,
    ) -> RecallResult:
        """Strict pattern completion from a masked latent cue.

        `mask` is True at coordinates that are known. Unknown coordinates
        should be zero-filled in `partial`; they are overwritten during
        settling anyway.
        """
        if partial.shape[-1] != self.store.mhn.dim:
            raise ValueError("partial cue must match the hippocampal key dimension")
        if mask.shape != partial.shape:
            raise ValueError("mask must have the same shape as the cue")
        return self._settle(
            partial.clone(),
            mask=mask.bool(),
            beta=beta,
            top_k=top_k,
            query=None,
            reinforce=reinforce,
            now=now,
        )

    def occlude(self, text: str, keep_fraction: float, *, seed: int = 0) -> tuple[Tensor, Tensor]:
        """Build a masked cue from text by keeping a random coordinate subset.

        Useful for evaluating completion quality: encode a known memory, hide
        most of its key, and check whether the network still settles on it.
        """
        key = self.encode_cue(text)
        g = torch.Generator().manual_seed(seed)
        keep = torch.rand(key.shape, generator=g) < keep_fraction
        return key * keep, keep

    def gist(
        self,
        query: str,
        *,
        top_k: int = 5,
        factor: float = 0.15,
        beta: float | None = None,
        reinforce: bool = False,
        **kwargs,
    ) -> RecallResult:
        """Deliberately low-beta recall: settle into a metastable mixture that
        summarises a family of related memories instead of one episode.

        `reinforce` defaults to False here, unlike `recall`: a schema-level read
        settles on a mixture, so whichever memory happens to top it is an
        artefact of the blend, not a memory the user actually recalled. Bumping
        its strength and resetting its decay clock would be wrong.
        """
        return self.recall(
            query,
            top_k=top_k,
            # An explicit beta wins over the factor; forwarding both used to
            # raise TypeError: got multiple values for 'beta'.
            beta=self.store.mhn.config.beta * factor if beta is None else beta,
            reinforce=reinforce,
            **kwargs,
        )
