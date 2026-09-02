"""Isotropic re-coding of an embedding space: centre + ZCA whitening.

Why this exists
---------------
A superposition of k unit vectors retains each component at cosine ~1/sqrt(k)
**only when the components are near-orthogonal**. Sentence embeddings are not:
measured here, BGE-small vectors for *unrelated* LoCoMo turns sit at mean
pairwise cosine **+0.649** -- a narrow cone in which every mixture of 4 looks
like every other mixture of 4. Centring and ZCA-whitening the same content
brings that to **-0.001**, and per-item recall of k memories superposed into ONE
384-d vector (decoded against 788 stored memories) changes accordingly:

    k     as shipped    whitened    random-vector ceiling
    2         0.915       1.000       1.000
    4         0.185       1.000       1.000
    8         0.062       0.999       1.000
    16        0.049       0.974       0.966
    32        0.070       0.892       0.850

Whitening recovers essentially the full theoretical capacity -- **when the
whitener is fit on the same vectors it then scores**, which is the transductive
case above. Fit on a disjoint corpus instead (review ticket 04: 5094 turns from
8 *other* LoCoMo conversations, applied unchanged to these 788), the same k=8
cell falls from 0.999 to 0.221 -- whitening still helps (unwhitened is 0.064),
but "recovers the full theoretical capacity" does not survive an honest fit.
See RESULTS.md Part V.2 for both numbers side by side, `fit`'s docstring for
why more held-out data does not close the gap, and
`ModernHopfieldNetwork.superpose` for the operation this feeds.

Why it is fitted, and not a function
------------------------------------
`experiments/superposition.py` whitens by re-fitting an SVD on whatever matrix
it is handed. That is fine for a one-shot measurement over a fixed corpus and
wrong for a library: a query whitened against its own singleton batch is not in
the same space as documents whitened against the corpus, so the cosines being
compared are meaningless. `Whitener` fits once and then applies the *same*
affine map -- mean, rotation, scale -- to documents and queries alike.

What it is not
--------------
Not a retrieval improvement. Measured on LoCoMo ranking, n=494 questions over
1451 turns, whitening the same BGE vectors moves hit@1 0.269 -> 0.302 (inside
this project's ~0.04 resolution limit, i.e. a tie) and *hurts* depth: hit@5
0.543 -> 0.536, hit@10 0.676 -> **0.628**. Anisotropy makes everything mildly
similar, which flatters fuzzy top-10 recall and destroys the component structure
superposition needs. Whiten for the substrate, not for the ranker.

Fitting once matters here too: refitting per conversation, as the prototype
does, costs a further 0.044 hit@5 and 0.077 hit@10 (0.492 / 0.551 -- the figures
published in RESULTS.md V.5, before this class existed), because a 369-turn fit
in 384 dimensions is rank-deficient. See `WhiteningConfig` and RESULTS.md V.6.

References: Mu & Viswanath 2018 ("all-but-the-top"); Su et al. 2021 ("whitening
sentence representations").
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import torch
from torch import Tensor

from .config import WhiteningConfig

__all__ = ["Whitener", "WhitenedEmbedder", "anisotropy"]


def anisotropy(
    x: Tensor, *, pairs: int = 20000, generator: torch.Generator | None = None
) -> float:
    """Mean cosine between random *distinct* pairs of rows. 0.0 is isotropic.

    The diagnostic that made the capacity problem visible: it is a property of
    the code, not of any query, so it can be read off a corpus before any
    retrieval is attempted. Reference points measured on 788 LoCoMo turns:
    BGE as shipped +0.649, the same vectors whitened -0.001, random unit
    vectors +0.000.

    Rows are unit-normalised here so the result is a cosine even if the caller
    passes unnormalised vectors. `generator` defaults to a fixed seed, because a
    diagnostic that moves between calls cannot be asserted on.
    """
    if x.dim() != 2:
        raise ValueError(f"expected a 2-D matrix of rows, got shape {tuple(x.shape)}")
    n = x.shape[0]
    if n < 2:
        raise ValueError("anisotropy needs at least 2 rows")
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    unit = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    i = torch.randint(0, n, (pairs,), generator=generator)
    j = torch.randint(0, n, (pairs,), generator=generator)
    keep = i != j
    if not bool(keep.any()):  # pathological: n==1 is already rejected above
        raise ValueError("no distinct pairs sampled; increase `pairs`")
    return float((unit[i[keep]] * unit[j[keep]]).sum(-1).mean())


class Whitener(torch.nn.Module):
    """Fitted centre + ZCA whitening, applied identically to documents and queries.

        transform(x) = normalise( (x - mu) V^T diag(1/s) V )

    where `V` (rows = right singular vectors of the centred corpus) and
    `s = max(sigma / sqrt(n), floor)` come from `fit`. ZCA rather than PCA
    whitening because ZCA is the whitening transform closest to the identity,
    so the whitened space stays as close to the original as isotropy allows --
    which matters when the vectors are still going to be read as "semantic".

    `floor` regularises the small singular values. Those are noise directions;
    without a floor they are divided by ~0 and end up dominating every cosine.
    1e-2 is the value the capacity measurements in RESULTS.md V.2 were made at.

    Rows are renormalised to the unit sphere afterwards, because that is what
    `HopfieldConfig.normalize_patterns` assumes and what makes the softmax
    logits comparable across memories.

    Fitting needs a corpus. It is a covariance estimate, so it is only as good
    as the sample: with fewer rows than dimensions the covariance is singular,
    only the fitted subspace is estimable, and `transform` projects onto it --
    see `fit`, which warns.
    """

    _mean: Tensor
    _rotation: Tensor
    _scale: Tensor

    def __init__(self, dim: int, config: WhiteningConfig | None = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.config = config or WhiteningConfig()
        if self.config.floor <= 0:
            raise ValueError(f"floor must be positive, got {self.config.floor}")
        # Empty until fitted. Shapes change on fit, which is exactly why the
        # state_dict hooks below are overridden rather than inherited.
        self.register_buffer("_mean", torch.empty(0))
        self.register_buffer("_rotation", torch.empty(0, self.dim))
        self.register_buffer("_scale", torch.empty(0))

    # ------------------------------------------------------------------ state

    @property
    def is_fitted(self) -> bool:
        return self._scale.numel() > 0

    @property
    def components(self) -> int:
        """Number of estimable directions kept, i.e. min(n_samples, dim)."""
        return int(self._scale.numel())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = f"components={self.components}" if self.is_fitted else "unfitted"
        return f"Whitener(dim={self.dim}, floor={self.config.floor:g}, {state})"

    # -------------------------------------------------------------- fit/apply

    def fit(self, x: Tensor) -> "Whitener":
        """Estimate the mean and covariance spectrum of a corpus matrix.

        Degenerate cases are rejected or narrowed rather than silently producing
        a transform that returns noise:

          * `n < 2` -- a single sample has no covariance at all; centring it
            gives the zero vector, and every subsequent `transform` would return
            an arbitrary direction. Raises.
          * `n < dim` -- the covariance is rank-deficient. Only the fitted
            subspace is estimable; `transform` projects onto it and discards the
            orthogonal complement, because scaling directions of measured-zero
            variance by 1/floor amplifies pure noise by 100x. Warns.
          * zero-variance directions -- floored, not inverted.
          * an entirely constant corpus -- raises; there is nothing to whiten.

        This warning guards sample *size*, not sample *representativeness*, and
        the two are not the same failure. RESULTS.md V.2 (review ticket 04)
        fits on a disjoint pool 13x past this threshold (5094 rows, d=384) drawn
        from different LoCoMo conversations than the vectors later transformed,
        and the residual anisotropy barely moves (+0.168 at n=64 to +0.167 at
        n=5094) -- while fitting on a same-conversation disjoint half at a third
        of the pool's size (n=394) gets anisotropy +0.003, matching an in-sample
        fit. A corpus that clears `n >= dim` by an order of magnitude can still
        leave most of the anisotropy gap unclosed if it does not resemble what
        gets transformed. Fit on a sample that represents the deployment
        traffic, not merely a large one.
        """
        x = x.detach()
        if x.dim() != 2:
            raise ValueError(f"expected a 2-D corpus matrix, got shape {tuple(x.shape)}")
        n, d = x.shape
        if d != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {d}")
        if not torch.isfinite(x).all():
            raise ValueError("corpus must be finite; got NaN or Inf")
        if n < 2:
            raise ValueError(
                f"cannot fit a whitener on {n} sample(s): centring leaves the zero "
                "vector and every transform would be an arbitrary direction"
            )
        if n < d:
            warnings.warn(
                f"whitening fitted on {n} samples in {d} dimensions: the covariance "
                f"is rank-deficient, so transform() projects onto the {n}-dimensional "
                "fitted subspace and discards the rest. Fit on more documents.",
                stacklevel=2,
            )

        mean = x.mean(dim=0)
        centred = x - mean
        _, sigma, vh = torch.linalg.svd(centred, full_matrices=False)
        std = sigma / math.sqrt(n)
        if float(std.max()) < 1e-8:
            raise ValueError(
                "corpus has no measurable variance in any direction; there is "
                "nothing to whiten (are all the rows identical?)"
            )
        self._mean = mean.clone()
        self._rotation = vh.clone()
        self._scale = std.clamp_min(self.config.floor)
        return self

    def transform(self, x: Tensor) -> Tensor:
        """Apply the fitted map. Accepts one vector (d,) or a batch (n, d).

        A row that lands exactly on the fitted mean (or lies entirely in the
        discarded complement) whitens to the zero vector; it is returned as
        zeros rather than as NaN. `ModernHopfieldNetwork.write` rejects such a
        row, which is the correct place for it to be caught -- a zero pattern
        scores log(w) against every query and hijacks retrieval.
        """
        if not self.is_fitted:
            raise RuntimeError("Whitener is not fitted; call fit(corpus) first")
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"expected (d,) or (n, d), got shape {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {x.shape[-1]}")
        centred = x.to(self._mean.dtype) - self._mean
        out = (centred @ self._rotation.T / self._scale) @ self._rotation
        if self.config.normalize:
            out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return out.squeeze(0) if single else out

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)

    def fit_transform(self, x: Tensor) -> Tensor:
        return self.fit(x).transform(x)

    # ------------------------------------------------------------ persistence

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        """Persist mean, rotation and scale as they are.

        Overridden for the same reason `ModernHopfieldNetwork` overrides it: the
        buffers are empty before `fit` and (components, dim) after, so the
        inherited in-place `copy_` load fails on a shape mismatch when restoring
        into a freshly constructed whitener -- which is the only way a whitener
        is ever restored.
        """
        for name in ("_mean", "_rotation", "_scale"):
            value = getattr(self, name)
            destination[prefix + name] = value if keep_vars else value.detach().clone()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        """Resize to the incoming fit instead of copying into fixed buffers."""
        keys = {name: prefix + name for name in ("_mean", "_rotation", "_scale")}
        incoming = {name: state_dict.get(key) for name, key in keys.items()}
        present = [name for name, value in incoming.items() if value is not None]
        if present:
            if len(present) != 3:
                error_msgs.append(
                    f"whitener state is incomplete: got {sorted(present)}, "
                    "expected _mean, _rotation and _scale together"
                )
                return
            rotation = incoming["_rotation"]
            if rotation.numel() and rotation.shape[-1] != self.dim:
                error_msgs.append(
                    f"_rotation has dim {rotation.shape[-1]}, expected {self.dim}"
                )
                return
            if incoming["_scale"].shape[0] != rotation.shape[0]:
                error_msgs.append(
                    f"_scale has {incoming['_scale'].shape[0]} entries but "
                    f"_rotation has {rotation.shape[0]} rows"
                )
                return
            for name, value in incoming.items():
                setattr(self, name, value.detach().clone())
        remaining = {k: v for k, v in state_dict.items() if k not in keys.values()}
        super()._load_from_state_dict(
            remaining, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        # Consumed above, so the base implementation -- which never saw them --
        # would otherwise report them missing under strict loading.
        for key in keys.values():
            if key in missing_keys:
                missing_keys.remove(key)


class WhitenedEmbedder:
    """An `Embedder` that whitens whatever the wrapped embedder produces.

    Exists so whitening can be turned on for a whole system by configuration
    (`MemorySystemConfig.whitening.enabled`) rather than by hand at every call
    site -- and, more importantly, so documents and queries cannot end up in
    different spaces, which is the failure the fitted `Whitener` prevents.

    Pass-through until fitted. `OrganizationalMemory.bootstrap` fits it on the
    bootstrap corpus, which is the only place a corpus exists; a system that
    enables whitening and never bootstraps therefore gets the unwhitened
    embedder, and `encode` warns once to say so rather than pretending.
    """

    def __init__(self, embedder, whitener: Whitener | None = None) -> None:
        self.embedder = embedder
        self.whitener = whitener or Whitener(embedder.dim)
        if self.whitener.dim != embedder.dim:
            raise ValueError(
                f"whitener dim {self.whitener.dim} != embedder dim {embedder.dim}"
            )
        self._warned = False

    @property
    def dim(self) -> int:
        return self.embedder.dim

    @property
    def rank(self):
        """Forwarded so `bootstrap`'s under-rank warning still fires."""
        return getattr(self.embedder, "rank", None)

    @property
    def is_fitted(self) -> bool:
        inner = getattr(self.embedder, "is_fitted", True)
        return bool(inner) and self.whitener.is_fitted

    def fit(self, texts: Sequence[str]) -> "WhitenedEmbedder":
        inner_fit = getattr(self.embedder, "fit", None)
        if callable(inner_fit) and not getattr(self.embedder, "is_fitted", True):
            inner_fit(texts)
        self.whitener.fit(self.embedder.encode(list(texts)))
        return self

    def _encode(self, texts: Sequence[str], *, raw: Tensor) -> Tensor:
        """Apply the fitted whitener to `raw`, or pass it through unfitted.

        Shared by `encode` and `encode_query` so the pass-through and
        warn-once behaviour is identical on both paths -- the query path used
        to not exist at all, which meant `getattr(embedder, "encode_query",
        embedder.encode)` call sites silently fell back to `encode` and lost
        the wrapped embedder's query prefix (e.g. BGE's instruction string)
        even though the baseline arm kept it. See `encode_query` below.
        """
        if not self.whitener.is_fitted:
            if not self._warned:
                warnings.warn(
                    "whitening is enabled but the whitener is not fitted; encodes "
                    "are passing through unwhitened. Call bootstrap(corpus) or "
                    "fit(corpus) first.",
                    stacklevel=2,
                )
                self._warned = True
            return raw
        return self.whitener.transform(raw)

    def encode(self, texts: Sequence[str]) -> Tensor:
        return self._encode(texts, raw=self.embedder.encode(texts))

    def encode_query(self, texts: Sequence[str]) -> Tensor:
        """Whiten the wrapped embedder's *query* encoding, not its document one.

        Delegates to `self.embedder.encode_query` when the wrapped embedder has
        one (e.g. BGE's instruction-prefixed query path), falling back to
        `encode` otherwise -- the same delegation `HybridEmbedder.encode_query`
        in `experiments/recall_ablation.py` uses. Without this method, every
        call-site pattern of the form
        `getattr(embedder, "encode_query", embedder.encode)` used `encode` for
        queries too, so a whitened BGE arm silently dropped the query prefix
        while the unwhitened baseline arm kept it -- every whitened-vs-shipped
        comparison was partly measuring that missing prefix rather than
        whitening. The *same* fitted whitener transform is applied here as in
        `encode`, because documents and queries must land in the same space.
        """
        inner_query = getattr(self.embedder, "encode_query", self.embedder.encode)
        return self._encode(texts, raw=inner_query(texts))
