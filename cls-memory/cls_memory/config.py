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

    input_dim: int = 384
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

    beta: float = 8.0
    """Inverse temperature. Ramsauer et al. (2021): high beta -> sharp, single
    pattern attractors (episodic recall); low beta -> metastable mixtures over
    many patterns (gist / schema recall). Also equals 1/sigma^2 of the
    equivalent diffusion noise level -- see cls_memory.energy."""
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

    mode: HippocampalKey = HippocampalKey.SEPARATED
    expansion_dim: int = 1024
    sparsity_k: int = 256


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

    This exists because decoder-only replay is a no-op for learning. Measured:
    `decode(latent)` of a stored anomaly had cosine 0.047 with the embedding it
    was supposed to be reinstating, and the cortex's surprise on its own decoder
    output was 0.0001 against 1.15 on the real memories -- so training on it is
    self-distillation with essentially zero gradient signal about the episodes,
    and pruning could never fire (0 of 36 released at any budget up to 150
    epochs).

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


@dataclass
class MemorySystemConfig:
    """Top-level container."""

    cortex: CortexConfig = field(default_factory=CortexConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    key: KeyConfig = field(default_factory=KeyConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    seed: int = 0
    device: str = "cpu"
