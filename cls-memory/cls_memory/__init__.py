"""CLS organizational memory -- a proof of concept.

Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995;
Kumaran, Hassabis & McClelland 2016 "extended model") applied to an
organisation's knowledge:

    neocortex    slow, overlapping, statistical  -> VAE over text embeddings
    hippocampus  fast, sparse, episodic          -> Modern Hopfield Network
    ingestion    novelty-gated routing between them
    consolidation replay + interleaved training, then prune what is now predictable
"""

from .config import (
    ConsolidationConfig,
    CortexConfig,
    DecayConfig,
    HopfieldConfig,
    IngestionConfig,
    KeyConfig,
    MemorySystemConfig,
    NoveltyConfig,
)
from .consolidation import ConsolidationEngine, ConsolidationReport
from .embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from .energy import (
    BasinReport,
    basin_depth,
    beta_to_sigma,
    denoise,
    langevin_replay,
    log_density,
    score,
    sigma_to_beta,
)
from .hippocampus import ModernHopfieldNetwork, RetrievalTrace
from .ingestion import IngestionAction, IngestionResult, SynapticIngestionPipeline
from .neocortex import NoveltyGate, SlowLearningNeocortex
from .pattern_separation import DentateGyrus, HippocampalKey, KeyEncoder
from .records import MemoryRecord, Persistence
from .retrieval import PatternCompleter, RecallResult, Recollection
from .store import MemoryStore, SweepReport
from .system import OrganizationalMemory

__version__ = "0.1.0"

__all__ = [
    "BasinReport",
    "ConsolidationConfig",
    "ConsolidationEngine",
    "ConsolidationReport",
    "CortexConfig",
    "DentateGyrus",
    "DecayConfig",
    "Embedder",
    "HashingEmbedder",
    "HippocampalKey",
    "HopfieldConfig",
    "IngestionAction",
    "IngestionConfig",
    "IngestionResult",
    "KeyConfig",
    "KeyEncoder",
    "MemoryRecord",
    "MemoryStore",
    "MemorySystemConfig",
    "ModernHopfieldNetwork",
    "NoveltyConfig",
    "NoveltyGate",
    "OrganizationalMemory",
    "PatternCompleter",
    "Persistence",
    "RecallResult",
    "Recollection",
    "RetrievalTrace",
    "SentenceTransformerEmbedder",
    "SlowLearningNeocortex",
    "SweepReport",
    "SynapticIngestionPipeline",
    "__version__",
    "basin_depth",
    "beta_to_sigma",
    "denoise",
    "langevin_replay",
    "log_density",
    "score",
    "sigma_to_beta",
]
