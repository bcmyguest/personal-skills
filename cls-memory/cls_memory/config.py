"""Central configuration for the CLS organizational memory system.

Every tunable lives here so a deployment can be described by a single object.
Defaults are chosen for a proof of concept on a few thousand documents, not
for production scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pattern_separation import HippocampalKey


@dataclass
class CortexConfig:
    """Slow-learning neocortex (VAE) hyperparameters."""

    input_dim: int = 1024
    """Embedding width, and so the fitted embedder's target rank.

    1024, not 384. Retrieval is embedder-bound and the dimension is the binding
    constraint on a truncated-SVD embedder: measured kNN recall@1 on LoCoMo runs
    0.077 (hashing-256), 0.174 (LSA-256), 0.255 (LSA-1024) against 0.320 for
    sparse TF-IDF with no reduction at all. Going wider costs memory linearly
    and buys most of that gap back. Lower it for small corpora -- the rank
    cannot exceed the number of documents, and bootstrap() warns when it."""
    hidden_dims: tuple[int, ...] = (256, 128)
    latent_dim: int = 64
    kl_weight: float = 0.01
    """beta in the beta-VAE objective.

    Much smaller than the textbook 1.0, and deliberately so. Embeddings are
    unit-norm, so the reconstruction term is O(1) while the KL term is
    O(latent_dim); at kl_weight=1 the KL dominates, the posterior collapses,
    and the decoder learns to emit the corpus mean. Reconstruction error then
    measures distance-from-mean rather than schema violation, and the novelty
    signal all but vanishes -- measured here as a 0.08 gap between on- and
    off-schema text at kl_weight=1 versus 0.53 at 0.01.

    Raise it only if you need a well-behaved generative latent; this system
    uses the VAE as a density/surprise estimator, not a sampler."""
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    """The schema is the slow half of CLS; undertraining it flattens the novelty
    signal, which silently turns the ingestion gate into a coin flip."""


@dataclass
class NoveltyConfig:
    """How surprise is measured and thresholded."""

    mode: str = "recon"
    """'recon' -> reconstruction error only (the classic novelty proxy).
    'elbo'   -> recon + KL, i.e. the variational bound on -log p(x). Strictly
                the better surprise estimate; 'recon' is the default because it
                is what the brief specifies and it is easier to interpret."""
    threshold: float | None = None
    """Absolute threshold. If None, the gate self-calibrates (below)."""
    calibrate: bool = True
    quantile: float = 0.85
    """When calibrating, an item is novel if its surprise exceeds this quantile
    of the recent surprise distribution. Self-calibration matters because the
    absolute scale of reconstruction error drifts as the cortex trains."""
    window: int = 512
    """Size of the rolling window used for quantile calibration."""
    warmup: int = 32
    """Items to observe before the calibrated gate is trusted. Below this count
    everything is treated as novel (an empty hippocampus should fill up)."""


@dataclass
class HopfieldConfig:
    """Fast-learning hippocampus (Modern Hopfield Network) hyperparameters."""

    beta: float = 128.0
    """Inverse temperature. High beta -> sharp single-pattern attractors
    (episodic recall); low beta -> metastable mixtures (gist / schema recall).
    Also equals 1/sigma^2 of the equivalent diffusion noise level (energy.py).

    128, not the 8.0 this shipped with. Retrieval needs beta*Delta_i to be
    comfortably large, and real embeddings are not well separated: measured DG
    keys for unrelated text sit at cosine ~0.71, giving Delta=0.29, so beta=8
    put beta*Delta at 2.3 and **every query settled onto one global mixture** --
    six distinct cues produced six settled states at pairwise cosine 1.0000,
    with `converged=True` reporting success. On LoCoMo that regime scores
    recall@1 0.004 against 0.172 at beta=128.

    Rule of thumb: keep beta * min_i effective_separation(i) above ~6. Lower
    beta deliberately, per query, when you want a gist (see gist())."""
    max_iter: int = 32
    tol: float = 1e-5
    """L2 change in the state below which the update is considered settled.
    The continuous MHN typically converges in one step; more iterations only
    matter for masked (partial-cue) retrieval."""
    normalize_patterns: bool = True
    """Project stored patterns onto the unit sphere. Required for the exact
    Gaussian-mixture / diffusion correspondence (see energy.py) and it keeps
    the softmax logits on a comparable scale across memories."""
    mixture_threshold: float = 0.5
    """If the largest softmax weight after settling is below this, the state is
    reported as a metastable mixture rather than a single retrieved memory."""


@dataclass
class KeyConfig:
    """Which vector the hippocampus stores, and how it is separated.

    The default deviates from the "store the VAE latent" specification on
    measured grounds -- see cls_memory.pattern_separation for the numbers.
    Set `mode=HippocampalKey.LATENT` to restore the original design.
    """

    mode: HippocampalKey = HippocampalKey.EMBEDDING
    """EMBEDDING, not SEPARATED. Dentate-gyrus separation was adopted because it
    beat the VAE latent on synthetic data (5/5 vs 2/5), and it still does -- but
    against the raw embedding on *real* text it is a net loss: LoCoMo recall@1
    0.121 separated vs 0.150 embedding at matched beta, and raising sparsity_k
    to 512 or 1024 does not recover it (0.117, 0.113). Sparsification discards
    lexical detail that real retrieval needs, which a 12-template synthetic
    corpus could not reveal. SEPARATED remains available and is the right choice
    when episodes are genuinely near-duplicate."""
    expansion_dim: int = 1024
    sparsity_k: int = 256


@dataclass
class WhiteningConfig:
    """Isotropic re-coding of the embedding space (cls_memory.whitening).

    Off by default, and the default is not a hedge -- whitening helps one thing
    and hurts another, measured on the same vectors:

      **Substrate (superposition).** A sum of k unit vectors keeps each
      component at cosine ~1/sqrt(k) only if the components are near-orthogonal.
      BGE-small vectors for unrelated LoCoMo turns sit at mean pairwise cosine
      +0.649; whitened, -0.001. Per-item recall of k memories held in ONE 384-d
      vector, decoded against 788 stored memories, goes from 0.185 to 1.000 at
      k=4 and from 0.062 to 0.999 at k=8 (RESULTS.md V.2). Without whitening
      superposition does not work at all beyond k=2.

      **Ranker (kNN retrieval).** The same transform on LoCoMo turn retrieval,
      n=494 questions over 1451 turns:

          BGE as shipped            @1 0.269   @5 0.543   @10 0.676
          whitened, fitted once     @1 0.302   @5 0.536   @10 0.628
          whitened, refit per conv  @1 0.304   @5 0.492   @10 0.551

      The hit@1 gain is +0.033, at this project's ~0.04 resolution limit, so
      read it as a tie. The depth loss is real and it is what this setting costs
      you. The third row is the prototype's refit-per-corpus behaviour and is
      the number RESULTS.md V.5 published: refitting on 369-663 turns in 384
      dimensions is rank-deficient, and roughly half the measured depth loss was
      that artifact rather than whitening. A `Whitener` fitted once on the whole
      corpus is both the correct usage and the cheaper trade (RESULTS.md V.6).

    Anisotropy makes everything mildly similar, which flatters fuzzy top-10
    recall and destroys the component structure superposition needs. The two
    uses want opposite geometries. **Whiten for the substrate, not the ranker.**

    Enabling this changes every number in RESULTS.md Parts I-IV, which were all
    measured with it off; that is why it is opt-in rather than a new default.
    Fitting needs a corpus, so it happens in `OrganizationalMemory.bootstrap`.
    """

    enabled: bool = False
    """Whiten every embedding the system produces, documents and queries alike.
    False keeps the geometry every published result here was measured on."""
    floor: float = 1e-2
    """Lower bound on a direction's standard deviation before inversion.

    The small singular values are noise directions; dividing by ~0 lets them
    dominate every cosine. 1e-2 is the value the V.2 capacity table was measured
    at -- k=8 at 0.999 per-item recall, matching the random-vector ceiling."""
    normalize: bool = True
    """Renormalise to the unit sphere after whitening. Required by
    `HopfieldConfig.normalize_patterns` and by the energy/diffusion
    correspondence in energy.py."""


@dataclass
class DecayConfig:
    """Temporal forgetting curve for episodic memories."""

    half_life_days: float = 30.0
    prune_below: float = 0.05
    """Salience floor. 0.05 with a 30-day half-life ~= 130 days of retention."""
    reinforce_on_recall: bool = True
    """Retrieval resets the decay clock (reconsolidation). Turning this off
    makes decay depend on creation time only."""
    reinforcement_gain: float = 0.25
    """Additive strength bonus per recall, capped at max_strength."""
    max_strength: float = 3.0


@dataclass
class IngestionConfig:
    """Synaptic ingestion routing rules."""

    duplicate_similarity: float = 0.97
    """Cosine similarity in latent space above which an incoming item is
    treated as a re-observation of an existing memory (reinforce, do not add)."""
    always_store_evergreen: bool = True
    """Evergreen facts bypass the novelty gate. A business rule that the cortex
    happens to predict must still be retrievable verbatim -- losing it to a
    lossy reconstruction is not an acceptable failure mode for a system of
    record. Set False for a purer CLS simulation."""


@dataclass
class ConsolidationConfig:
    """Systems consolidation: hippocampus -> neocortex transfer."""

    replay_batch: int = 64
    replay_steps: int = 8
    replay_step_size: float = 0.05
    """Langevin step size used when sampling the hippocampal energy landscape."""
    episodic_ratio: float = 0.125
    """Fraction of the replay batch drawn from the *stored embeddings* rather
    than from the cortex's own decoder.

    This exists because decoder-only replay is a no-op for learning. The cortex's
    surprise on its own decoder output is ~100x lower than on the real memories
    (0.0086 vs 0.902 in one measurement, 0.0001 vs 1.15 in another) -- training
    on it is self-distillation with essentially zero gradient signal about the
    episodes, and pruning could never fire (0 of 36 released at any budget up to
    150 epochs). The cosine between `decode(latent)` and the embedding it should
    reinstate is likewise poor, but its exact value is configuration-specific
    (0.047 and 0.358 measured under different settings) and should not be quoted
    as a general figure.

    The value is a measured operating point on a real trade-off, not a guess.
    Episodic replay teaches the episodes and, unchecked, overwrites the schema
    while doing it; the generated share is what holds the schema in place. Full
    benchmark, 30 episodic memories, drift measured against naive training on
    off-schema data:

        ratio   drift reduction   pruned @20ep   pruned @40ep
        0.000            55.3%              0              0
        0.125            54.9%              6             16
        0.250            50.3%             15             21
        0.500            46.0%             23             29

    0.125 is the knee: the 0.4-point drift cost is inside run-to-run noise (the
    same measurement has moved 57.1 -> 56.1 -> 55.3 across reruns), and it is
    the smallest share that makes the loop close at all. Raise it if draining
    the hippocampus matters more than schema stability in your deployment, and
    expect to pay roughly one point of drift protection per 2-3 extra memories
    released."""
    replay_sigma: float = 0.05
    """Diffusion noise level for embedding-space replay, i.e. sigma in the
    Gaussian mixture the hippocampus encodes (see energy.py); the landscape is
    sampled at beta = 1 / sigma^2.

    Deliberately NOT the hippocampal `HopfieldConfig.beta`. Beta is chosen for
    retrieval sharpness over 2048-d sparse keys; reused as a noise level over
    256-d unit-norm embeddings it gives sigma = 1/sqrt(32) = 0.177, whose
    samples have norm 3.41 and cosine 0.295 to the nearest stored memory --
    the usual high-dimensional Gaussian-mixture failure the seed noise in
    `langevin_replay` already works around. Measured cosine to the nearest
    stored memory: 0.939 at sigma=0.02, 0.737 at 0.05, 0.481 at 0.1. 0.05
    smooths enough that the cortex learns the neighbourhood of each episode
    rather than the point."""
    interleave_ratio: float = 0.5
    """Fraction of each consolidation batch drawn from replay rather than new
    data. Interleaving is what prevents catastrophic forgetting in CLS."""
    relative_drop: float = 0.5
    """Prune a memory once the cortex reconstructs it at less than this
    fraction of its surprise at ingestion. A relative test, because an absolute
    quantile threshold drifts with the ingestion stream and fires without any
    consolidation having happened.

    0.5 survives the check that matters, now that surprise actually moves.
    After a consolidation pass the memories it released reconstructed at cosine
    0.843 to their own embedding and at worst 0.496 absolute surprise, against
    a mean of 0.798 for routine documents under the same cortex -- i.e. every
    released memory was predicted at least as well as the schema's own training
    data, which is the strongest statement "the schema now covers it" can
    reasonably mean. Loosening to 0.7 releases 26 at cosine 0.739; tightening
    to 0.2 releases 4 at 0.892."""
    prune_predicted: bool = True
    """After consolidation, drop episodic memories the cortex now reconstructs
    below threshold -- the hippocampal trace has been absorbed by the schema."""
    protect_evergreen: bool = True
    """Never consolidation-prune evergreen records, even once predictable."""


def _check_range(name: str, value: float, lo: float, hi: float, *, inclusive=True) -> None:
    ok = lo <= value <= hi if inclusive else lo < value < hi
    if not ok:
        bounds = f"[{lo}, {hi}]" if inclusive else f"({lo}, {hi})"
        raise ValueError(f"{name} must be in {bounds}, got {value}")


@dataclass
class MemorySystemConfig:
    """Top-level container."""

    cortex: CortexConfig = field(default_factory=CortexConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    key: KeyConfig = field(default_factory=KeyConfig)
    whitening: WhiteningConfig = field(default_factory=WhiteningConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate at construction, not at first use.

        Several of these previously surfaced only deep inside a run -- an
        out-of-range quantile raised from torch *after* the cortex had trained,
        and relative_drop >= 1 silently reintroduced a defect this project had
        already fixed once (it pruned 2 of 3 memories from an untouched cortex).
        """
        _check_range("novelty.quantile", self.novelty.quantile, 0.0, 1.0)
        _check_range("hopfield.beta", self.hopfield.beta, 0.0, float("inf"), inclusive=False)
        _check_range(
            "hopfield.mixture_threshold", self.hopfield.mixture_threshold, 0.0, 1.0
        )
        _check_range(
            "ingestion.duplicate_similarity",
            self.ingestion.duplicate_similarity, 0.0, 1.0, inclusive=False,
        )
        _check_range(
            "consolidation.relative_drop",
            self.consolidation.relative_drop, 0.0, 1.0, inclusive=False,
        )
        _check_range(
            "consolidation.replay_sigma",
            self.consolidation.replay_sigma, 0.0, float("inf"), inclusive=False,
        )
        _check_range(
            "consolidation.episodic_ratio", self.consolidation.episodic_ratio, 0.0, 1.0
        )
        _check_range("decay.half_life_days", self.decay.half_life_days, 0.0,
                     float("inf"), inclusive=False)
        # A floor of 0 divides the noise directions by their measured standard
        # deviation, which for the near-null ones is ~0: the whitened vector
        # then points wherever the smallest eigenvector happens to point.
        _check_range("whitening.floor", self.whitening.floor, 0.0, float("inf"),
                     inclusive=False)
