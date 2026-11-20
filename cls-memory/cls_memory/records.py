"""Memory records and the evergreen / temporal distinction.

The salience of a record is a scalar in (0, max_strength] that gates how
strongly it competes during retrieval. Episodic records decay exponentially;
evergreen records sit permanently at the ceiling.

Salience enters the Hopfield retrieval as a log-prior on the softmax logits
(see hippocampus.ModernHopfieldNetwork), which is equivalent to giving the
memory a mixing weight in the Gaussian mixture the network encodes. That is
strictly better than shrinking the pattern vector itself, which would distort
the geometry of the attractor and corrupt the energy function.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

import torch

SECONDS_PER_DAY = 86_400.0


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Persistence(str, Enum):
    """Which forgetting regime a memory belongs to."""

    EVERGREEN = "evergreen"
    """Business rules, policies, definitions, org structure. Timeless: exempt
    from the forgetting curve and (by default) from consolidation pruning."""

    TEMPORAL = "temporal"
    """Episodic logs: incidents, meetings, deploys, chat. Decays with a
    configurable half-life; a stale incident should stop competing with a
    fresh one during recall."""


@dataclass
class MemoryRecord:
    """One item in the organizational memory."""

    text: str
    embedding: torch.Tensor
    """Original encoder-space embedding (d_input,). Kept so consolidation can
    replay the true surface form into the cortex, not just its latent."""
    latent: torch.Tensor
    """VAE latent (d_latent,). Always retained -- it is the cortex's view of
    the item, used for schema-level comparisons."""
    key: torch.Tensor | None = None
    """The vector actually stored in the Hopfield network. Defaults to the
    latent when not supplied (the originally specified design); normally the
    pattern-separated code -- see cls_memory.pattern_separation."""
    persistence: Persistence = Persistence.TEMPORAL
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=utcnow)
    last_reinforced_at: datetime = field(default_factory=utcnow)
    access_count: int = 0
    strength: float = 1.0
    """Multiplicative durability, raised by repeated recall/re-observation."""
    novelty: float = 0.0
    """Surprise at ingestion time, for auditing what the schema failed to
    predict."""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.key is None:
            # clone, or the key would alias the latent and mutating one would
            # silently corrupt the other
            self.key = self.latent.clone()

    @property
    def is_evergreen(self) -> bool:
        return self.persistence is Persistence.EVERGREEN

    def age_days(self, now: datetime | None = None, *, from_reinforcement: bool = True) -> float:
        now = now or utcnow()
        anchor = self.last_reinforced_at if from_reinforcement else self.created_at
        return max(0.0, (now - anchor).total_seconds() / SECONDS_PER_DAY)

    def salience(
        self,
        now: datetime | None = None,
        *,
        half_life_days: float = 30.0,
        from_reinforcement: bool = True,
        max_strength: float = 3.0,
    ) -> float:
        """Current retrieval weight.

        Evergreen:  w = max_strength   (the ceiling, so no amount of recall
                                        lets an episode outrank a business rule)
        Temporal:   w = clamp(strength) * 2 ** (-age_days / half_life)

        The exponential form is the standard Ebbinghaus forgetting curve; the
        half-life parameterisation means `half_life_days=30` literally halves
        a memory's competitive weight every 30 days.
        """
        if self.is_evergreen:
            return max_strength
        age = self.age_days(now, from_reinforcement=from_reinforcement)
        decay = math.pow(2.0, -age / max(half_life_days, 1e-9))
        return min(self.strength, max_strength) * decay

    def reinforce(
        self,
        now: datetime | None = None,
        *,
        gain: float = 0.25,
        max_strength: float = 3.0,
        reset_clock: bool = True,
    ) -> None:
        """Reconsolidation: recalling a memory makes it durable again."""
        self.access_count += 1
        self.strength = min(self.strength + gain, max_strength)
        if reset_clock:
            self.last_reinforced_at = now or utcnow()

    def age_by(self, days: float) -> None:
        """Test/demo helper: backdate the record to simulate the passage of time."""
        delta = timedelta(days=days)
        self.created_at -= delta
        self.last_reinforced_at -= delta

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        head = self.text if len(self.text) <= 48 else self.text[:45] + "..."
        return f"MemoryRecord({self.persistence.value}, {head!r}, id={self.id[:8]})"
