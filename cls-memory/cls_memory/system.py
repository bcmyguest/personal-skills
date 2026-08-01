"""Facade wiring the five components into one organisational memory system.

    embedder -> neocortex (VAE) -> novelty gate -> hippocampus (MHN)
                     ^                                   |
                     +--------- consolidation <----------+

Use the components directly when you need control; use this class when you
want the whole loop with one object.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import torch
from torch import Tensor

from .config import MemorySystemConfig
from .consolidation import ConsolidationEngine, ConsolidationReport
from .embeddings import Embedder, HashingEmbedder
from .ingestion import IngestionResult, SynapticIngestionPipeline
from .neocortex import NoveltyGate, SlowLearningNeocortex
from .pattern_separation import KeyEncoder
from .records import MemoryRecord, Persistence
from .retrieval import PatternCompleter, RecallResult
from .store import MemoryStore, SweepReport


@dataclass
class BootstrapReport:
    corpus_size: int
    final_loss: float
    novelty_threshold: float


class OrganizationalMemory:
    """CLS-based memory: slow semantic schema + fast episodic store."""

    def __init__(
        self,
        config: MemorySystemConfig | None = None,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config or MemorySystemConfig()
        torch.manual_seed(self.config.seed)

        self.embedder = embedder or HashingEmbedder(
            dim=self.config.cortex.input_dim, seed=self.config.seed
        )
        if self.embedder.dim != self.config.cortex.input_dim:
            # Keep the cortex honest about its actual input width.
            self.config.cortex.input_dim = self.embedder.dim

        self.cortex = SlowLearningNeocortex(self.config.cortex)
        self.gate = NoveltyGate(self.config.novelty)
        self.key_encoder = KeyEncoder(
            self.config.key.mode,
            embedding_dim=self.config.cortex.input_dim,
            latent_dim=self.config.cortex.latent_dim,
            expansion_dim=self.config.key.expansion_dim,
            sparsity_k=self.config.key.sparsity_k,
            seed=self.config.seed,
        )
        self.store = MemoryStore(
            self.key_encoder.dim,
            hopfield=self.config.hopfield,
            decay=self.config.decay,
        )
        self.ingestion = SynapticIngestionPipeline(
            self.embedder,
            self.cortex,
            self.store,
            self.key_encoder,
            gate=self.gate,
            novelty=self.config.novelty,
            config=self.config.ingestion,
        )
        self.retrieval = PatternCompleter(
            self.embedder, self.cortex, self.store, self.key_encoder
        )
        self.consolidation = ConsolidationEngine(
            self.cortex,
            self.store,
            gate=self.gate,
            config=self.config.consolidation,
            reencode_key=self._reencode_key,
        )

    def _reencode_key(self, record: MemoryRecord) -> Tensor:
        """Recompute a record's hippocampal key from its stored embedding.

        Used after consolidation trains the cortex. For SEPARATED/EMBEDDING
        keys this is a no-op in effect (they do not depend on the cortex); for
        LATENT keys it is what stops training from destroying the memory.
        """
        latent = self.cortex.latent(record.embedding)
        record.latent = latent
        return self.key_encoder(record.embedding, latent)

    # ------------------------------------------------------------- lifecycle

    def bootstrap(
        self, corpus: Sequence[str], *, epochs: int | None = None, verbose: bool = False
    ) -> BootstrapReport:
        """Train the schema on historical text, then calibrate the gate.

        Order matters: calibrating before training would measure surprise
        against an untrained cortex, where everything looks equally novel.
        """
        x = self.embedder.encode(list(corpus))
        history = self.cortex.fit(x, epochs=epochs, verbose=verbose)
        self.ingestion.calibrate(corpus)
        return BootstrapReport(
            corpus_size=len(corpus),
            final_loss=history[-1] if history else float("nan"),
            novelty_threshold=self.gate.threshold,
        )

    # ------------------------------------------------------------------ write

    def ingest(
        self,
        text: str,
        *,
        persistence: Persistence = Persistence.TEMPORAL,
        metadata: dict | None = None,
        now: datetime | None = None,
    ) -> IngestionResult:
        return self.ingestion.ingest(
            text, persistence=persistence, metadata=metadata, now=now
        )

    def remember_rule(self, text: str, **kwargs) -> IngestionResult:
        """Ingest an evergreen business rule (no forgetting curve)."""
        return self.ingest(text, persistence=Persistence.EVERGREEN, **kwargs)

    def log_event(self, text: str, **kwargs) -> IngestionResult:
        """Ingest a temporal episodic event (30-day half-life by default)."""
        return self.ingest(text, persistence=Persistence.TEMPORAL, **kwargs)

    # ------------------------------------------------------------------- read

    def recall(self, query: str, **kwargs) -> RecallResult:
        return self.retrieval.recall(query, **kwargs)

    def gist(self, query: str, **kwargs) -> RecallResult:
        return self.retrieval.gist(query, **kwargs)

    def complete(self, partial: Tensor, mask: Tensor, **kwargs) -> RecallResult:
        return self.retrieval.complete(partial, mask, **kwargs)

    # ------------------------------------------------------------ maintenance

    def sleep(
        self,
        new_texts: Sequence[str] | None = None,
        *,
        epochs: int = 5,
        now: datetime | None = None,
    ) -> ConsolidationReport:
        """Run one consolidation cycle (replay + interleaved training + prune)."""
        new_data = self.embedder.encode(list(new_texts)) if new_texts else None
        return self.consolidation.consolidate(new_data, epochs=epochs, now=now)

    def sweep(self, now: datetime | None = None) -> SweepReport:
        """Apply the forgetting curve without touching the cortex."""
        return self.store.sweep(now)

    # ------------------------------------------------------------------- info

    def stats(self, now: datetime | None = None) -> dict:
        s = self.store.stats(now)
        s["novelty_threshold"] = self.gate.threshold
        s["gate_observations"] = self.gate.observed
        s["beta"] = self.config.hopfield.beta
        return s

    @property
    def records(self) -> list[MemoryRecord]:
        return self.store.records

    def __len__(self) -> int:
        return len(self.store)
