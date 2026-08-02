"""Fast-learning hippocampus: a continuous Modern Hopfield Network.

Implements the energy function and update rule of Ramsauer et al. (2021),
"Hopfield Networks is All You Need" (arXiv:2008.02217), extended with a
per-pattern log-prior so that salience (the forgetting curve) can modulate
retrieval without distorting the stored geometry.

Notation
--------
X in R^{N x d}   stored patterns (rows x_i), unit-normalised
xi in R^d        query / state
beta > 0         inverse temperature
w_i              salience of pattern i, normalised to sum to 1
M = max_i ||x_i||

Energy (Ramsauer eq. 4, with a prior):

    E(xi) = -beta^-1 * logsumexp_i( beta * x_i . xi - beta * ||x_i||^2 / 2
                                    + log w_i )
            + 0.5 * xi . xi

The per-pattern -beta*||x_i||^2/2 inside the logsumexp replaces the global
+0.5*M^2 term of the paper. For unit-norm patterns the two are identical: a
constant inside logsumexp factors straight out, and -(-beta/2)/beta = +0.5 =
0.5*M^2. For patterns that are *not* unit-norm they differ, and the per-pattern
form is the correct one -- it is what makes exp(-beta*E) the Gaussian mixture
centred on the patterns rather than a norm-tilted version of it, so the score,
the Tweedie denoiser and log_density stay mutually consistent at any pattern
scale. See energy.py; consolidation deliberately runs unnormalised.

With uniform w_i = 1/N and unit-norm patterns this reduces exactly to Ramsauer's

    E(xi) = -lse(beta, X xi) + 0.5 xi.xi + beta^-1 log N + 0.5 M^2

Update rule (Ramsauer eq. 3) is the concave-convex procedure applied to E,
and therefore decreases the energy monotonically:

    xi_new = X^T softmax(beta * X xi + log w)

Key properties this module exposes:
  * `energy`      - the scalar above; a negative log-density up to a constant.
  * `step`        - one CCCP update = one attention head = one denoising step.
  * `retrieve`    - iterate to a fixed point, optionally clamping known
                    coordinates of a partial cue (pattern completion).
  * `separation`  - Delta_i, which controls whether pattern i is a genuine
                    fixed point with exponentially small retrieval error.

Storage is a growable buffer; removal compacts the tensor. That is O(N*d) per
delete, which is fine at proof-of-concept scale and keeps `patterns` a dense
contiguous tensor for fast matmuls.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import HopfieldConfig


@dataclass
class RetrievalTrace:
    """Everything observable about one settling run."""

    state: Tensor
    """Settled state xi* (d,)."""
    weights: Tensor
    """Softmax weights over stored patterns at xi* (N,)."""
    energy: float
    """E(xi*) -- lower means a deeper basin, i.e. better-remembered."""
    energy_path: list[float]
    iterations: int
    converged: bool
    is_mixture: bool
    """True if no single pattern dominates: a metastable state averaging over
    several memories. Useful on purpose -- it is schema-level 'gist' recall."""

    @property
    def top_weight(self) -> float:
        return float(self.weights.max()) if self.weights.numel() else 0.0


def _normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class ModernHopfieldNetwork(torch.nn.Module):
    """Continuous-state associative memory over a mutable pattern set.

    Pure tensor mechanics only: it knows nothing about text, records or time.
    Bookkeeping lives in `cls_memory.store.MemoryStore`.
    """

    patterns: Tensor
    log_prior: Tensor

    def __init__(self, dim: int, config: HopfieldConfig | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.config = config or HopfieldConfig()
        self.register_buffer("_patterns", torch.empty(0, dim))
        self.register_buffer("_log_prior", torch.empty(0))
        self._n_used = 0

    # ------------------------------------------------------------------ state

    @property
    def patterns(self) -> Tensor:
        """The live pattern rows.

        A view into the capacity buffer, so it is exactly `len(self)` rows
        regardless of allocated capacity. **Any `write` or `remove` may
        invalidate it**: a write that triggers growth reallocates, and a view
        held across that silently stops aliasing the buffer. Clone if you need
        to keep it."""
        return self._patterns[: self._n_used]

    @patterns.setter
    def patterns(self, value: Tensor) -> None:
        # Sets all three fields together. The old setter bumped _n_used without
        # touching _log_prior, so `net.patterns = X` with a different row count
        # left the two views disagreeing -- silently in one direction, as an
        # opaque broadcast error in the other.
        self._patterns = value
        self._n_used = int(value.shape[0])
        if self._log_prior.shape[0] < self._n_used:
            grown = torch.zeros(self._n_used, dtype=self._log_prior.dtype)
            grown[: self._log_prior.shape[0]] = self._log_prior
            self._log_prior = grown

    @property
    def log_prior(self) -> Tensor:
        return self._log_prior[: self._n_used]

    @log_prior.setter
    def log_prior(self, value: Tensor) -> None:
        if value.shape[0] < self._n_used:
            raise ValueError(
                f"log_prior has {value.shape[0]} rows but {self._n_used} "
                "patterns are live"
            )
        self._log_prior = value

    def __len__(self) -> int:
        return int(self._n_used)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def write(self, x: Tensor, log_prior: Tensor | float = 0.0) -> Tensor:
        """Store one or more patterns. Returns the new row indices.

        'Fast learning' in the CLS sense: a single exposure, no gradient steps,
        no interference with what is already stored.
        """
        x = x.detach().to(self.patterns.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected patterns of dim {self.dim}, got {x.shape[-1]}")
        # One NaN or Inf row poisons energy, attention, step, retrieve,
        # log_density and replay for the WHOLE store, for every query,
        # permanently and silently. Reject at the boundary; the check is one
        # pass over the new rows.
        if not torch.isfinite(x).all():
            raise ValueError("patterns must be finite; got NaN or Inf")
        if (x.norm(dim=-1) < 1e-8).any():
            raise ValueError(
                "zero-norm pattern: such a row has logit log(w) regardless of the "
                "query and outranks any real memory below cosine 0.5, silently "
                "hijacking retrieval. Reject the source text instead."
            )
        if self.config.normalize_patterns:
            x = _normalize(x)

        if isinstance(log_prior, (int, float)):
            lp = torch.full((x.shape[0],), float(log_prior), dtype=self.log_prior.dtype)
        else:
            lp = log_prior.detach().reshape(-1).to(self.log_prior.dtype)
            if lp.shape[0] != x.shape[0]:
                raise ValueError("log_prior must have one entry per pattern")
        # set_log_prior checked this; write did not, leaving the same
        # store-poisoning defect open one argument over.
        if not torch.isfinite(lp).all():
            raise ValueError("log_prior must be finite; got NaN or Inf")

        # Geometric growth, not a fresh torch.cat per write. Reallocating the
        # whole buffer on every single-pattern write made ingestion O(N^2):
        # measured at d=2048, doubling N cost ~3.6x the time, which extrapolates
        # to hours of pure buffer copying at 100k memories.
        start = self._n_used
        needed = start + x.shape[0]
        # Capacity is the ALLOCATED buffer, not the live view: reading it from
        # `self.patterns` (length n_used) made every write look like it needed
        # to grow, which is the O(N^2) this replaces.
        if needed > self._patterns.shape[0]:
            capacity = max(8, self._patterns.shape[0])
            while capacity < needed:
                capacity *= 2
            grown = torch.zeros(
                capacity, self.dim, dtype=self.patterns.dtype, device=self.patterns.device
            )
            grown[:start] = self.patterns[:start]
            self.patterns = grown
            grown_prior = torch.zeros(
                capacity, dtype=self.log_prior.dtype, device=self.log_prior.device
            )
            grown_prior[:start] = self.log_prior[:start]
            self.log_prior = grown_prior
        self._patterns[start:needed] = x
        self._log_prior[start:needed] = lp
        self._n_used = needed
        return torch.arange(start, needed)

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        """Persist the live rows only.

        `_patterns`/`_log_prior` hold *capacity*, and `_n_used` is a plain int
        that state_dict cannot carry. Saving the raw buffers stored the zero
        padding and dropped the only thing recording how much of it was real:
        loading into a network whose capacity happened to match restored the
        padding as live memories, and those zero rows took **100% of the
        attention mass** (a zero row scores log(w) for any query, while a real
        unit-norm row scores beta*(cos - 0.5)). Loading into a fresh network
        failed outright on a shape mismatch, strict=False included.
        """
        destination[prefix + "_patterns"] = (
            self.patterns if keep_vars else self.patterns.detach().clone()
        )
        destination[prefix + "_log_prior"] = (
            self.log_prior if keep_vars else self.log_prior.detach().clone()
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        """Resize to the incoming rows and restore the live-row counter."""
        pattern_key = prefix + "_patterns"
        prior_key = prefix + "_log_prior"
        remaining = {
            k: v for k, v in state_dict.items() if k not in (pattern_key, prior_key)
        }
        patterns = state_dict.get(pattern_key)
        log_prior = state_dict.get(prior_key)
        if patterns is not None:
            if patterns.shape[-1] != self.dim:
                error_msgs.append(
                    f"_patterns has dim {patterns.shape[-1]}, expected {self.dim}"
                )
                return
            self._patterns = patterns.detach().clone()
            self._n_used = int(patterns.shape[0])
            self._log_prior = (
                log_prior.detach().clone()
                if log_prior is not None
                else torch.zeros(self._n_used, dtype=self._patterns.dtype)
            )
        super()._load_from_state_dict(
            remaining, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        # We consumed these above, so the base implementation -- which never saw
        # them -- would otherwise report them missing under strict loading.
        if patterns is not None:
            for key in (pattern_key, prior_key):
                if key in missing_keys:
                    missing_keys.remove(key)

    def remove(self, indices: Tensor | list[int]) -> None:
        """Forget patterns by row index (compacts the buffers)."""
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long)
        if indices.numel() == 0:
            return
        keep = torch.ones(len(self), dtype=torch.bool)
        keep[indices.to(torch.long)] = False
        # Compacts to exactly n_used, so the next write reallocates. remove is
        # already O(N*d), so this is a ~2-3x constant on prune-then-ingest
        # cycles, not an asymptotic change.
        kept_patterns = self.patterns[keep].contiguous()
        kept_prior = self.log_prior[keep].contiguous()
        self._patterns = kept_patterns
        self._log_prior = kept_prior
        self._n_used = int(kept_patterns.shape[0])

    def set_log_prior(self, log_prior: Tensor) -> None:
        """Refresh all salience weights at once (called by the decay sweep)."""
        log_prior = log_prior.reshape(-1).to(self.log_prior.dtype)
        if not torch.isfinite(log_prior).all():
            raise ValueError("log_prior must be finite; got NaN or Inf")
        if log_prior.shape[0] != len(self):
            raise ValueError("log_prior length must match the number of patterns")
        self._log_prior[: self._n_used] = log_prior.detach()

    # ------------------------------------------------------------- mechanics

    def _beta(self, beta: float | None) -> float:
        return float(self.config.beta if beta is None else beta)

    def logits(self, xi: Tensor, beta: float | None = None) -> Tensor:
        """beta*(X xi - ||x_i||^2/2) + log w, with log w a normalised prior.

        Renormalising the prior (subtracting its logsumexp) is what makes the
        uniform case reproduce Ramsauer's `beta^-1 log N` term exactly, and it
        keeps the energy comparable across different memory-set sizes.

        The -||x_i||^2/2 term is a no-op for unit-norm patterns and is what
        keeps the diffusion identities exact when they are not normalised.
        """
        beta = self._beta(beta)
        prior = self.log_prior - torch.logsumexp(self.log_prior, dim=0)
        half_sq = 0.5 * (self.patterns * self.patterns).sum(-1)
        return beta * (xi @ self.patterns.T - half_sq) + prior

    def attention(self, xi: Tensor, beta: float | None = None) -> Tensor:
        """softmax over stored patterns -- the association strengths."""
        return torch.softmax(self.logits(xi, beta), dim=-1)

    def step(self, xi: Tensor, beta: float | None = None) -> Tensor:
        """One CCCP update: xi <- X^T softmax(beta X xi + log w).

        Identical in form to a transformer attention head with Q=xi, K=V=X,
        and identical to one posterior-mean denoising step of a diffusion model
        over the stored patterns (see cls_memory.energy).
        """
        return self.attention(xi, beta) @ self.patterns

    def energy(self, xi: Tensor, beta: float | None = None) -> Tensor:
        """E(xi) from the module docstring. Lower = deeper basin."""
        if self.is_empty:
            raise RuntimeError("energy is undefined for an empty hippocampus")
        beta = self._beta(beta)
        lse = torch.logsumexp(self.logits(xi, beta), dim=-1) / beta
        return -lse + 0.5 * (xi * xi).sum(-1)

    def grad_energy(self, xi: Tensor, beta: float | None = None) -> Tensor:
        """Analytic gradient: dE/dxi = xi - X^T softmax(...).

        Closed form, so no autograd is needed anywhere in retrieval or replay.
        """
        return xi - self.step(xi, beta)

    def separation(self, index: int) -> float:
        """Delta_i = x_i.x_i - max_{j != i} x_i.x_j.

        Ramsauer's separation, computed exactly as the paper defines it. The
        paper's guarantee -- retrieval error for pattern i exponentially small
        in beta*Delta_i -- assumes a *uniform* prior, so it does NOT transfer
        to this network once salience varies. Use `effective_separation` for a
        diagnostic that accounts for the prior.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"pattern index {index} out of range for {len(self)}")
        if len(self) < 2:
            return float("inf")
        sims = self.patterns[index] @ self.patterns.T
        self_sim = float(sims[index])
        others = torch.cat([sims[:index], sims[index + 1 :]])
        return self_sim - float(others.max())

    def effective_separation(self, index: int, beta: float | None = None) -> float:
        """Separation including the salience prior.

            Delta_i + (log w_i - max_{j != i} log w_j) / beta

        This is the quantity that actually governs whether pattern i is a fixed
        point of *this* network. A decayed memory can be geometrically well
        separated and still lose retrieval to a fresher neighbour, which the
        raw `separation` would not reveal.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"pattern index {index} out of range for {len(self)}")
        if len(self) < 2:
            return float("inf")
        beta = self._beta(beta)
        lp = self.log_prior
        others = torch.cat([lp[:index], lp[index + 1 :]])
        return self.separation(index) + (float(lp[index]) - float(others.max())) / beta

    # ------------------------------------------------------------- retrieval

    def retrieve(
        self,
        xi: Tensor,
        *,
        mask: Tensor | None = None,
        beta: float | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> RetrievalTrace:
        """Settle a (possibly partial) cue onto an attractor.

        `mask` is a boolean tensor over the state dimensions, True where the
        cue is *known*. Known coordinates are clamped after every update and
        only the unknown ones are allowed to move -- this is pattern completion
        in the strict sense: the network fills in the missing coordinates from
        the stored associations rather than being handed them.

        With `mask=None` the whole (noisy) cue is free to move, which is the
        right behaviour for a text query: the query embedding is itself a
        corrupted version of the pattern, not a subset of its coordinates.
        """
        if self.is_empty:
            raise RuntimeError("cannot retrieve from an empty hippocampus")

        max_iter = self.config.max_iter if max_iter is None else max_iter
        tol = self.config.tol if tol is None else tol
        state = xi.detach().clone().to(self.patterns.dtype)
        cue = state.clone()

        path: list[float] = [float(self.energy(state, beta))]
        converged = False
        iterations = 0
        for iterations in range(1, max_iter + 1):
            nxt = self.step(state, beta)
            if mask is not None:
                nxt = torch.where(mask, cue, nxt)
            delta = float((nxt - state).norm())
            state = nxt
            path.append(float(self.energy(state, beta)))
            if delta < tol:
                converged = True
                break

        weights = self.attention(state, beta)
        return RetrievalTrace(
            state=state,
            weights=weights,
            energy=path[-1],
            energy_path=path,
            iterations=iterations,
            converged=converged,
            is_mixture=float(weights.max()) < self.config.mixture_threshold,
        )

    # NOTE: a coarse-to-fine beta schedule ("annealed retrieval") is the obvious
    # thing to try given the diffusion equivalence, and it was implemented and
    # measured here. It does not work for cue-driven recall: low-beta passes
    # pull the state toward cluster means, discarding exactly the cue detail
    # that distinguishes memories within a theme. On clustered patterns it
    # retrieved the right memory 17/40 times against 31/40 for the plain
    # one-step update. Annealing belongs on the *sampling* side, where it is
    # used -- see energy.langevin_replay.
