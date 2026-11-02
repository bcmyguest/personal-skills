"""Tests for the embedders, especially the fitted LSA one.

The embedder turned out to dominate real-data retrieval quality — measured on
LoCoMo, swapping hashing for LSA moved recall@5 from 0.045 to 0.174 — so its
contract is worth pinning down.
"""

from __future__ import annotations

import pytest
import torch

from cls_memory import HashingEmbedder, LatentSemanticEmbedder

CORPUS = [
    "the shard rebalancer corrupted the orders index during failover",
    "the shard rebalancer was restarted after the orders index failure",
    "a contractor deleted the staging kubernetes namespace by hand",
    "the staging kubernetes namespace was restored from backup",
    "the payment provider rotated credentials without notice",
    "payment provider credentials were rotated again this quarter",
    "deployment of the billing service completed without incident",
    "deployment of the checkout service completed without incident",
    "nightly batch job finished successfully in the eu-west region",
    "nightly batch job finished successfully in the us-east region",
    "weekly report delivered to the finance team on schedule",
    "weekly report delivered to the operations team on schedule",
]


def test_hashing_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=64, seed=0)
    a = embedder.encode(CORPUS[:3])
    b = embedder.encode(CORPUS[:3])
    assert torch.allclose(a, b)
    assert torch.allclose(a.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_lsa_requires_fitting_first():
    embedder = LatentSemanticEmbedder(dim=16)
    assert not embedder.is_fitted
    with pytest.raises(RuntimeError):
        embedder.encode(["anything"])


def test_lsa_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        LatentSemanticEmbedder(dim=8).fit([])


def test_lsa_honours_requested_dim_even_on_a_tiny_corpus():
    """A corpus-dependent embedding width would make the cortex's input_dim
    depend on how much text happened to exist at bootstrap."""
    embedder = LatentSemanticEmbedder(dim=64, min_df=1).fit(CORPUS)
    assert embedder.dim == 64
    assert embedder.rank <= len(CORPUS)
    vectors = embedder.encode(CORPUS[:4])
    assert vectors.shape == (4, 64)
    # Padded coordinates are structurally zero, not noise.
    assert torch.allclose(vectors[:, embedder.rank :], torch.zeros(4, 64 - embedder.rank))


def test_lsa_is_deterministic_and_normalised():
    a = LatentSemanticEmbedder(dim=16, min_df=1, seed=3).fit(CORPUS)
    b = LatentSemanticEmbedder(dim=16, min_df=1, seed=3).fit(CORPUS)
    va, vb = a.encode(CORPUS[:5]), b.encode(CORPUS[:5])
    assert torch.allclose(va, vb, atol=1e-5)
    assert torch.allclose(va.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_lsa_places_topically_related_text_closer_than_unrelated():
    """The substantive claim: LSA should recover topical structure that pure
    lexical hashing cannot, without either pair sharing a rare exact token."""
    embedder = LatentSemanticEmbedder(dim=32, min_df=1, seed=0).fit(CORPUS)
    vectors = embedder.encode(CORPUS)
    related = float(vectors[0] @ vectors[1])  # both about the shard rebalancer
    unrelated = float(vectors[0] @ vectors[10])  # rebalancer vs weekly report
    assert related > unrelated


def test_lsa_handles_text_with_no_known_terms():
    """Out-of-vocabulary input must not produce NaNs downstream."""
    embedder = LatentSemanticEmbedder(dim=16, min_df=1, seed=0).fit(CORPUS)
    vector = embedder.encode(["中文 テキスト"])
    assert vector.shape == (1, 16)
    assert torch.isfinite(vector).all()


def test_unfitted_embedder_is_fitted_by_bootstrap():
    """A fittable embedder must be fitted on the bootstrap corpus, not lazily
    on whatever single document happens to arrive first."""
    from cls_memory import CortexConfig, MemorySystemConfig, OrganizationalMemory

    embedder = LatentSemanticEmbedder(dim=16, min_df=1, seed=0)
    system = OrganizationalMemory(
        MemorySystemConfig(
            cortex=CortexConfig(
                input_dim=16, hidden_dims=(16,), latent_dim=8, epochs=5, batch_size=4
            ),
            seed=0,
        ),
        embedder=embedder,
    )
    assert not embedder.is_fitted
    system.bootstrap(CORPUS)
    assert embedder.is_fitted
