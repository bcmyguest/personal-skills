"""Bookkeeping layer: keeps MemoryRecords in sync with Hopfield pattern rows.

The MHN is a dense tensor addressed by row index; the rest of the system wants
stable record ids. This class owns that mapping, plus the decay sweep that
pushes current salience into the network's log-prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Iterator

import torch
from torch import Tensor

from .config import DecayConfig, HopfieldConfig
from .hippocampus import ModernHopfieldNetwork
from .records import MemoryRecord, Persistence, utcnow

MIN_LOG_PRIOR = -30.0
"""Floor for log(salience) so a fully decayed record cannot produce -inf and
poison the softmax normaliser before it is pruned."""


@dataclass
class SweepReport:
    """Result of one maintenance pass."""

    evaluated: int
    pruned_decayed: int
    pruned_ids: list[str]
    min_salience: float
    mean_salience: float


class MemoryStore:
    """Records + hippocampal index, kept consistent."""

    def __init__(
        self,
        key_dim: int,
        *,
        hopfield: HopfieldConfig | None = None,
        decay: DecayConfig | None = None,
    ) -> None:
        self.decay = decay or DecayConfig()
        self.mhn = ModernHopfieldNetwork(key_dim, hopfield)
        self._records: list[MemoryRecord] = []
        self._index: dict[str, int] = {}

    # ------------------------------------------------------------ container

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[MemoryRecord]:
        # Snapshot: `remove` rebinds `self._records`, so a live iterator would
        # silently walk the stale list rather than raising the way mutating a
        # dict or list during iteration would.
        return iter(list(self._records))

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._index

    @property
    def records(self) -> list[MemoryRecord]:
        return list(self._records)

    def get(self, record_id: str) -> MemoryRecord:
        return self._records[self._index[record_id]]

    def row_of(self, record_id: str) -> int:
        return self._index[record_id]

    def record_at(self, row: int) -> MemoryRecord:
        return self._records[row]

    def filter(self, predicate: Callable[[MemoryRecord], bool]) -> list[MemoryRecord]:
        return [r for r in self._records if predicate(r)]

    # --------------------------------------------------------------- writes

    def add(self, record: MemoryRecord, *, now: datetime | None = None) -> MemoryRecord:
        """Single-shot write into the hippocampus."""
        if record.id in self._index:
            raise ValueError(f"record id {record.id} is already stored")
        if record.key.shape[-1] != self.mhn.dim:
            raise ValueError(
                f"key dim {record.key.shape[-1]} != hippocampal dim {self.mhn.dim}"
            )
        self.mhn.write(record.key, self._log_prior(record, now))
        # The network normalises on write; mirror that back onto the record so
        # `keys()` and `mhn.patterns` cannot drift apart.
        record.key = self.mhn.patterns[-1].clone()
        self._index[record.id] = len(self._records)
        self._records.append(record)
        return record

    def reindex(self, key_of) -> None:
        """Recompute every stored key and rebuild the pattern matrix.

        Required after anything that changes the encoder -- notably training
        the cortex, which silently invalidates latent-derived keys. `key_of`
        takes a MemoryRecord and returns its new key.

        Atomic: the new patterns are built and validated before anything is
        removed, and the old buffers are restored if the write fails. Removing
        first left the store with records but zero patterns on any bad key --
        permanently unusable, since `refresh_priors`, `sweep` and `recall` all
        then raise, and `consolidate` calls this automatically.
        """
        if not self._records:
            return
        keys = torch.stack([key_of(r) for r in self._records])
        if keys.shape[-1] != self.mhn.dim:
            raise ValueError(
                f"reindex produced keys of dim {keys.shape[-1]}, "
                f"expected {self.mhn.dim}"
            )
        priors = self.mhn.log_prior.clone()
        snapshot = self.mhn.patterns.clone()
        self.mhn.remove(torch.arange(len(self._records)))
        try:
            self.mhn.write(keys, priors)
        except Exception:
            self.mhn.patterns = snapshot
            self.mhn.log_prior = priors
            raise
        for row, record in enumerate(self._records):
            record.key = self.mhn.patterns[row].clone()

    def assert_consistent(self) -> None:
        """Records, index and pattern rows must agree.

        Cheap enough to call on every mutating path. Desynchronisation is
        silent otherwise: `record_at(row)` happily returns the wrong record
        after a direct `store.mhn.remove(...)`, and nothing raises.
        """
        if len(self._records) != len(self.mhn):
            raise RuntimeError(
                f"store desynchronised: {len(self._records)} records but "
                f"{len(self.mhn)} patterns. Mutate the store, never store.mhn."
            )
        if len(self._index) != len(self._records):
            raise RuntimeError("store index desynchronised from records")
        # Cardinality alone missed the worst case: an index with the right
        # number of entries but wrong values makes record_at(row) return the
        # wrong record, which is exactly what this is meant to prevent.
        for row, record in enumerate(self._records):
            if self._index.get(record.id) != row:
                raise RuntimeError(
                    f"store index points {record.id} at row "
                    f"{self._index.get(record.id)}, expected {row}"
                )

    def remove(self, record_ids: Iterable[str]) -> list[str]:
        """Forget records by id. Returns the ids actually removed."""
        # Deduplicate: remove(["a", "a"]) previously reported two removals.
        seen: set[str] = set()
        ids = [
            rid
            for rid in record_ids
            if rid in self._index and not (rid in seen or seen.add(rid))
        ]
        if not ids:
            return []
        rows = sorted(self._index[rid] for rid in ids)
        self.mhn.remove(torch.tensor(rows, dtype=torch.long))
        drop = set(rows)
        self._records = [r for i, r in enumerate(self._records) if i not in drop]
        self._index = {r.id: i for i, r in enumerate(self._records)}
        return ids

    # ---------------------------------------------------------------- decay

    def _log_prior(self, record: MemoryRecord, now: datetime | None = None) -> float:
        w = record.salience(
            now,
            half_life_days=self.decay.half_life_days,
            from_reinforcement=self.decay.reinforce_on_recall,
            max_strength=self.decay.max_strength,
        )
        return max(math.log(max(w, 1e-12)), MIN_LOG_PRIOR)

    def salience_vector(self, now: datetime | None = None) -> Tensor:
        now = now or utcnow()
        return torch.tensor(
            [
                r.salience(
                    now,
                    half_life_days=self.decay.half_life_days,
                    from_reinforcement=self.decay.reinforce_on_recall,
                    max_strength=self.decay.max_strength,
                )
                for r in self._records
            ],
            dtype=torch.float32,
        )

    def refresh_priors(self, now: datetime | None = None) -> None:
        """Recompute every log-prior from the current clock.

        Cheap (one tensor write), so call it before any retrieval that needs
        time-accurate weighting rather than trying to decay incrementally.
        """
        if not self._records:
            return
        self.assert_consistent()
        w = self.salience_vector(now)
        self.mhn.set_log_prior(torch.log(w.clamp_min(1e-12)).clamp_min(MIN_LOG_PRIOR))

    def sweep(self, now: datetime | None = None) -> SweepReport:
        """Apply the forgetting curve and prune what has decayed past the floor.

        Evergreen records always have salience 1 and can never be pruned here.
        """
        now = now or utcnow()
        self.assert_consistent()
        if not self._records:
            # 0.0, matching stats(); the two used to disagree on the sentinel
            # for "no memories", reporting 1.0 here and 0.0 there.
            return SweepReport(0, 0, [], 0.0, 0.0)

        w = self.salience_vector(now)
        doomed = [
            r.id
            for r, wi in zip(self._records, w.tolist())
            if not r.is_evergreen and wi < self.decay.prune_below
        ]
        evaluated = len(self._records)
        pruned = self.remove(doomed)
        self.refresh_priors(now)

        live = self.salience_vector(now)
        return SweepReport(
            evaluated=evaluated,
            pruned_decayed=len(pruned),
            pruned_ids=pruned,
            min_salience=float(live.min()) if live.numel() else 0.0,
            mean_salience=float(live.mean()) if live.numel() else 0.0,
        )

    # ----------------------------------------------------------------- misc

    def stats(self, now: datetime | None = None) -> dict:
        w = self.salience_vector(now)
        n_ever = sum(1 for r in self._records if r.is_evergreen)
        return {
            "total": len(self._records),
            "evergreen": n_ever,
            "temporal": len(self._records) - n_ever,
            "mean_salience": float(w.mean()) if w.numel() else 0.0,
            "mean_access_count": (
                sum(r.access_count for r in self._records) / len(self._records)
                if self._records
                else 0.0
            ),
        }

    def keys(self) -> Tensor:
        if not self._records:
            return torch.empty(0, self.mhn.dim)
        return torch.stack([r.key for r in self._records])

    def latents(self) -> Tensor:
        if not self._records:
            return torch.empty(0, 0)
        return torch.stack([r.latent for r in self._records])

    def embeddings(self) -> Tensor:
        if not self._records:
            return torch.empty(0, 0)
        return torch.stack([r.embedding for r in self._records])

    def by_persistence(self, persistence: Persistence) -> list[MemoryRecord]:
        return self.filter(lambda r: r.persistence is persistence)
