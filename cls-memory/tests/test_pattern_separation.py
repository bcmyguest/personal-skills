"""Tests for the dentate-gyrus pattern separator and key routing.

The load-bearing claim is that separation actually decorrelates: inputs that
are close in embedding space must be further apart in key space, otherwise the
hippocampus fuses distinct episodes into one attractor.
"""

from __future__ import annotations

import pytest
import torch

from cls_memory import (
    CortexConfig,
    DentateGyrus,
    HippocampalKey,
    KeyConfig,
    MemorySystemConfig,
    NoveltyConfig,
    OrganizationalMemory,
)
from cls_memory.pattern_separation import KeyEncoder
from tests.test_memory_system import CORPUS


def test_output_is_sparse_and_normalised():
    dg = DentateGyrus(64, expansion_dim=512, sparsity_k=32)
    x = torch.randn(8, 64)
    h = dg(x)
    assert h.shape == (8, 512)
    assert torch.allclose(h.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert int((h != 0).sum(dim=-1).max()) <= 32


def test_single_vector_keeps_its_shape():
    dg = DentateGyrus(64, expansion_dim=256, sparsity_k=16)
    assert dg(torch.randn(64)).shape == (256,)


def _pair_at_cosine(dim: int, cosine: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two unit vectors at an exact cosine. Building the pair by adding noise
    is misleading in high dimensions -- a 0.25-scaled Gaussian perturbation of
    a unit vector has norm ~2 and lands near cosine 0.3, not 0.97."""
    g = torch.Generator().manual_seed(seed)
    a = torch.nn.functional.normalize(torch.randn(dim, generator=g), dim=-1)
    r = torch.randn(dim, generator=g)
    perp = torch.nn.functional.normalize(r - (r @ a) * a, dim=-1)
    sine = (1.0 - cosine**2) ** 0.5
    return a, cosine * a + sine * perp


def test_separation_decorrelates_similar_inputs():
    """The whole point: similar inputs must become less similar as keys."""
    dg = DentateGyrus(64, expansion_dim=1024, sparsity_k=64)
    a, b = _pair_at_cosine(64, 0.95)

    input_sim = float(a @ b)
    key_sim = float(dg(a) @ dg(b))
    assert input_sim == pytest.approx(0.95, abs=1e-5)
    assert key_sim < input_sim


def test_sparser_codes_separate_more():
    a, b = _pair_at_cosine(64, 0.9)
    sims = []
    for k in (512, 128, 32):
        dg = DentateGyrus(64, expansion_dim=1024, sparsity_k=k)
        sims.append(float(dg(a) @ dg(b)))
    assert sims[0] > sims[1] > sims[2]


def test_projection_is_fixed_across_calls():
    """A drifting projection would silently invalidate every stored key."""
    dg = DentateGyrus(32, expansion_dim=128, sparsity_k=16, seed=3)
    x = torch.randn(32)
    assert torch.allclose(dg(x), dg(x))
    assert torch.allclose(dg.projection, DentateGyrus(32, 128, 16, seed=3).projection)


def test_rejects_k_larger_than_expansion():
    with pytest.raises(ValueError):
        DentateGyrus(16, expansion_dim=32, sparsity_k=64)


@pytest.mark.parametrize(
    "mode,expected_dim",
    [
        (HippocampalKey.SEPARATED, 256),
        (HippocampalKey.EMBEDDING, 64),
        (HippocampalKey.LATENT, 16),
    ],
)
def test_key_encoder_dimensions(mode, expected_dim):
    enc = KeyEncoder(
        mode, embedding_dim=64, latent_dim=16, expansion_dim=256, sparsity_k=32
    )
    assert enc.dim == expected_dim
    key = enc(torch.randn(64), torch.randn(16))
    assert key.shape == (expected_dim,)
    assert float(key.norm()) == pytest.approx(1.0, abs=1e-5)


def _system(mode: HippocampalKey) -> OrganizationalMemory:
    return OrganizationalMemory(
        MemorySystemConfig(
            cortex=CortexConfig(
                input_dim=64,
                hidden_dims=(48,),
                latent_dim=16,
                epochs=300,
                batch_size=8,
                learning_rate=3e-3,
            ),
            novelty=NoveltyConfig(quantile=0.8, warmup=4, window=64),
            key=KeyConfig(mode=mode, expansion_dim=1024, sparsity_k=256),
            seed=0,
        )
    )


EVENTS = [
    "the shard rebalancer corrupted the orders index during failover",
    "a contractor deleted the staging kubernetes namespace",
    "the payment provider rotated credentials without notice",
    "an unfamiliar vendor invoice arrived from a shell company",
    "the billing service double charged 200 customers overnight",
]
QUERIES = [
    "shard rebalancer orders index",
    "contractor staging kubernetes",
    "payment provider credentials",
    "vendor invoice shell company",
    "billing double charge customers",
]


def _recall_accuracy(mode: HippocampalKey) -> tuple[int, float]:
    system = _system(mode)
    system.bootstrap(CORPUS)
    records = [system.log_event(t).record for t in EVENTS]
    assert all(r is not None for r in records)

    correct = sum(
        int(system.recall(q, reinforce=False).top.record.id == r.id)
        for q, r in zip(QUERIES, records)
    )
    patterns = system.store.mhn.patterns
    off_diagonal = float(
        (patterns @ patterns.T - torch.eye(len(patterns))).abs().max()
    )
    return correct, off_diagonal


def test_separated_key_beats_the_raw_latent():
    """Regression guard on the finding that motivated this module: storing the
    VAE latent fuses distinct novel episodes, because the cortex has no
    resolution outside the routine text it was trained on."""
    sep_correct, sep_offdiag = _recall_accuracy(HippocampalKey.SEPARATED)
    lat_correct, lat_offdiag = _recall_accuracy(HippocampalKey.LATENT)

    assert sep_correct > lat_correct
    assert sep_offdiag < lat_offdiag
    assert sep_correct == len(QUERIES)


def test_latent_mode_still_runs():
    """The originally specified design must remain selectable and functional,
    even though it retrieves less accurately."""
    system = _system(HippocampalKey.LATENT)
    system.bootstrap(CORPUS)
    assert system.store.mhn.dim == 16
    record = system.log_event(EVENTS[0]).record
    assert record is not None
    assert torch.allclose(record.key, record.latent / record.latent.norm(), atol=1e-5)
    assert system.recall(QUERIES[0], reinforce=False).top is not None
