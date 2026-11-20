"""Synaptic ingestion: route incoming text by how surprising the schema finds it.

    embed -> cortex surprise -> gate
                                 |
             surprise >  theta --+--> encode latent, write to hippocampus
             surprise <= theta --+--> PREDICTED: do not store (and reinforce
                                      the schema-adjacent memory if one exists)

The design intent is that the hippocampus only ever holds what the neocortex
cannot regenerate. Everything predictable is left to the schema, which is what
keeps the fast store small enough to stay a genuine attractor network rather
than a vector database with extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence

import torch
from torch import Tensor

from .config import IngestionConfig, NoveltyConfig
from .embeddings import Embedder
from .neocortex import NoveltyGate, SlowLearningNeocortex
from .pattern_separation import KeyEncoder
from .records import MemoryRecord, Persistence, utcnow
from .store import MemoryStore


class IngestionAction(str, Enum):
    STORED = "stored"
    """Novel: written to the hippocampus."""
    PREDICTED = "predicted"
    """Reconstructed below threshold -- the schema already covers it, so it is
    pruned rather than stored."""
    REINFORCED = "reinforced"
    """A near-duplicate of an existing memory: strengthen that one instead of
    adding a second copy."""
    STORED_EVERGREEN = "stored_evergreen"
    """Bypassed the gate because it is a business rule."""
    REJECTED = "rejected"
    """The text produced a zero embedding -- no in-vocabulary terms at all
    (empty, whitespace, punctuation, emoji, or entirely out-of-vocabulary
    script). Such a key cannot be compared to anything: deduplication never
    fires on it, and in the Hopfield logits it scores log(w) regardless of the
    query, beating any real memory below cosine 0.5. One junk record was
    measured hijacking retrieval for unrelated queries at weight 0.985."""


@dataclass
class IngestionResult:
    action: IngestionAction
    text: str
    novelty: float
    threshold: float
    record: MemoryRecord | None = None
    duplicate_of: str | None = None
    similarity: float | None = None

    @property
    def was_stored(self) -> bool:
        return self.action in (IngestionAction.STORED, IngestionAction.STORED_EVERGREEN)


class SynapticIngestionPipeline:
    """Novelty-gated write path into the memory system."""

    def __init__(
        self,
        embedder: Embedder,
        cortex: SlowLearningNeocortex,
        store: MemoryStore,
        key_encoder: KeyEncoder,
        *,
        gate: NoveltyGate | None = None,
        novelty: NoveltyConfig | None = None,
        config: IngestionConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.cortex = cortex
        self.store = store
        self.key_encoder = key_encoder
        self.novelty = novelty or NoveltyConfig()
        self.gate = gate or NoveltyGate(self.novelty)
        self.config = config or IngestionConfig()

    # ---------------------------------------------------------------- helpers

    def _encode(self, text: str) -> tuple[Tensor, Tensor, Tensor, float]:
        embedding = self.embedder.encode([text])[0]
        surprise = float(self.cortex.surprise(embedding, self.novelty.mode)[0])
        latent = self.cortex.latent(embedding)
        key = self.key_encoder(embedding, latent)
        return embedding, latent, key, surprise

    def _nearest(self, key: Tensor) -> tuple[str | None, float]:
        if len(self.store) == 0:
            return None, -1.0
        sims = self.store.mhn.patterns @ key
        idx = int(sims.argmax())
        return self.store.record_at(idx).id, float(sims[idx])

    # ---------------------------------------------------------------- ingest

    def ingest(
        self,
        text: str,
        *,
        persistence: Persistence = Persistence.TEMPORAL,
        metadata: dict | None = None,
        now: datetime | None = None,
    ) -> IngestionResult:
        now = now or utcnow()
        embedding, latent, key, surprise = self._encode(text)

        # Reject before anything else: a degenerate key poisons dedup and
        # retrieval, and the evergreen bypass would otherwise walk it straight
        # past the gate.
        # The EMBEDDING must be checked too, not only the key. Under
        # HippocampalKey.LATENT a zero embedding still yields a unit-norm key,
        # so the record stored -- and then consolidation's
        # _embedding_landscape rejected the zero embedding, bricking sleep()
        # permanently. The evergreen bypass walked it straight past the gate.
        degenerate = (
            not torch.isfinite(key).all()
            or float(key.norm()) < 1e-8
            or not torch.isfinite(embedding).all()
            or float(embedding.norm()) < 1e-8
        )
        if degenerate:
            return IngestionResult(
                action=IngestionAction.REJECTED,
                text=text,
                novelty=surprise,
                threshold=self.gate.threshold,
            )

        is_novel, threshold = self.gate(surprise)

        # 1. Near-duplicate check runs first: re-observing a known fact is a
        #    reinforcement event regardless of what the novelty gate thinks.
        dup_id, sim = self._nearest(key)
        if dup_id is not None and sim >= self.config.duplicate_similarity:
            record = self.store.get(dup_id)
            record.reinforce(
                now,
                gain=self.store.decay.reinforcement_gain,
                max_strength=self.store.decay.max_strength,
                reset_clock=self.store.decay.reinforce_on_recall,
            )
            self.store.refresh_priors(now)
            return IngestionResult(
                action=IngestionAction.REINFORCED,
                text=text,
                novelty=surprise,
                threshold=threshold,
                record=record,
                duplicate_of=dup_id,
                similarity=sim,
            )

        # 2. Evergreen bypass. A business rule must be retrievable verbatim even
        #    if the cortex finds it unsurprising -- see IngestionConfig.
        evergreen_bypass = (
            persistence is Persistence.EVERGREEN and self.config.always_store_evergreen
        )

        # 3. Predicted by the schema -> prune (never enters the hippocampus).
        if not is_novel and not evergreen_bypass:
            return IngestionResult(
                action=IngestionAction.PREDICTED,
                text=text,
                novelty=surprise,
                threshold=threshold,
                duplicate_of=dup_id,
                similarity=sim,
            )

        record = MemoryRecord(
            text=text,
            embedding=embedding,
            latent=latent,
            key=key,
            persistence=persistence,
            created_at=now,
            last_reinforced_at=now,
            novelty=surprise,
            metadata=metadata or {},
        )
        self.store.add(record, now=now)
        action = (
            IngestionAction.STORED_EVERGREEN
            if (persistence is Persistence.EVERGREEN and not is_novel)
            else IngestionAction.STORED
        )
        return IngestionResult(
            action=action,
            text=text,
            novelty=surprise,
            threshold=threshold,
            record=record,
            similarity=sim if dup_id else None,
        )

    def ingest_many(
        self,
        texts: Sequence[str],
        *,
        persistence: Persistence = Persistence.TEMPORAL,
        now: datetime | None = None,
    ) -> list[IngestionResult]:
        return [self.ingest(t, persistence=persistence, now=now) for t in texts]

    # ------------------------------------------------------------ calibration

    def calibrate(self, texts: Sequence[str]) -> Tensor:
        """Prime the novelty gate on a corpus without storing anything.

        Run this on the historical corpus the cortex was trained on, so the
        first live item is judged against a populated surprise distribution
        instead of an empty window.
        """
        embeddings = self.embedder.encode(list(texts))
        scores = self.cortex.surprise(embeddings, self.novelty.mode)
        self.gate.prime(scores)
        return scores
