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
    interleave_ratio: float = 0.5
    """Fraction of each consolidation batch drawn from replay rather than new
    data. Interleaving is what prevents catastrophic forgetting in CLS."""
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
