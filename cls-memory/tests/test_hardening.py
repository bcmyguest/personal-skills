"""Regression guards for the hardening review findings.

Each test reproduces a defect that was verified before it was fixed. The bias
throughout is toward *silent* failures — a crash announces itself, a store that
quietly returns the wrong memory does not.
"""

from __future__ import annotations

import time

import pytest
import torch

from cls_memory import (
    ConsolidationConfig,
    CortexConfig,
    HashingEmbedder,
    HippocampalKey,
    HopfieldConfig,
    IngestionAction,
    KeyConfig,
    LatentSemanticEmbedder,
    MemoryRecord,
    MemoryStore,
    MemorySystemConfig,
    NoveltyConfig,
    OrganizationalMemory,
    Persistence,
)
from cls_memory.hippocampus import ModernHopfieldNetwork
from tests.test_memory_system import CORPUS, make_system


# ------------------------------------------------------- non-finite / degenerate


def test_write_rejects_non_finite_patterns():
    """One NaN row used to poison energy, attention, retrieve, log_density and
    replay for the entire store, for every query, permanently and silently."""
    net = ModernHopfieldNetwork(4, HopfieldConfig())
    net.write(torch.eye(4))
    for bad in (torch.tensor([[float("nan")] * 4]), torch.tensor([[float("inf")] * 4])):
        with pytest.raises(ValueError, match="finite"):
            net.write(bad)
    assert len(net) == 4
    assert torch.isfinite(net.step(torch.randn(4))).all()


def test_write_rejects_zero_norm_patterns():
    net = ModernHopfieldNetwork(4, HopfieldConfig())
    with pytest.raises(ValueError, match="zero-norm"):
        net.write(torch.zeros(1, 4))


def test_set_log_prior_rejects_non_finite():
    net = ModernHopfieldNetwork(4, HopfieldConfig())
    net.write(torch.eye(4))
    with pytest.raises(ValueError, match="finite"):
        net.set_log_prior(torch.tensor([0.0, float("nan"), 0.0, 0.0]))


@pytest.mark.parametrize("text", ["", "   \t ", "!!! ???", "🎉🎉🎉", "你好世界"])
def test_degenerate_text_is_rejected_not_stored(text):
    """Such text produces a zero embedding. Stored, its logit is log(w)
    regardless of the query, so it outranks any real memory below cosine 0.5 —
    one junk record was measured hijacking retrieval at weight 0.985. Dedup
    cannot catch it either, since cosine between two zero vectors is 0."""
    system = make_system()
    system.bootstrap(CORPUS)
    before = len(system)

    result = system.log_event(text)
    assert result.action is IngestionAction.REJECTED
    assert result.record is None
    assert len(system) == before

    # The evergreen bypass must not walk it past the check either.
    assert system.remember_rule(text).action is IngestionAction.REJECTED
    assert len(system) == before


def test_junk_cannot_hijack_retrieval():
    system = make_system()
    system.bootstrap(CORPUS)
    target = system.log_event(
        "the shard rebalancer corrupted the orders index during failover"
    ).record
    assert target is not None
    system.log_event("🎉🎉🎉")

    result = system.recall("shard rebalancer orders index", reinforce=False)
    assert result.top.record.id == target.id


# ----------------------------------------------------------------- state integrity


def test_reindex_is_atomic():
    """Removing before writing left records with zero patterns on any bad key —
    permanently unusable, and reachable from consolidate() via a user-supplied
    reencode_key."""
    store = MemoryStore(4)
    for i in range(3):
        store.add(
            MemoryRecord(
                text=f"r{i}", embedding=torch.zeros(4), latent=torch.eye(4)[i].clone()
            )
        )
    before = store.mhn.patterns.clone()

    with pytest.raises(ValueError):
        store.reindex(lambda r: torch.ones(8))  # wrong width

    assert len(store) == 3
    assert len(store.mhn) == 3
    assert torch.allclose(store.mhn.patterns, before)
    store.refresh_priors()  # must not raise
    store.assert_consistent()


def test_assert_consistent_catches_desynchronisation():
    store = MemoryStore(4)
    for i in range(3):
        store.add(
            MemoryRecord(
                text=f"r{i}", embedding=torch.zeros(4), latent=torch.eye(4)[i].clone()
            )
        )
    store.mhn.remove([0])  # reaching past the store, which owns the invariant
    with pytest.raises(RuntimeError, match="desynchronised"):
        store.assert_consistent()


def test_remove_deduplicates_ids():
    store = MemoryStore(4)
    record = MemoryRecord(text="a", embedding=torch.zeros(4), latent=torch.eye(4)[0])
    store.add(record)
    assert store.remove([record.id, record.id]) == [record.id]


def test_iteration_is_snapshotted_against_concurrent_removal():
    store = MemoryStore(4)
    records = []
    for i in range(3):
        r = MemoryRecord(
            text=f"r{i}", embedding=torch.zeros(4), latent=torch.eye(4)[i].clone()
        )
        store.add(r)
        records.append(r)
    seen = []
    for record in store:
        seen.append(record.id)
        if len(seen) == 1:
            store.remove([records[2].id])
    assert len(seen) == 3  # the snapshot is walked in full, not silently truncated


def test_empty_store_sentinels_agree():
    store = MemoryStore(4)
    report = store.sweep()
    assert report.mean_salience == store.stats()["mean_salience"] == 0.0


# ------------------------------------------------------------------ reproducibility


def test_constructing_a_system_does_not_perturb_another():
    """Seeding used to be process-global, so building a second system silently
    changed an existing one's training and replay."""

    def build() -> OrganizationalMemory:
        return OrganizationalMemory(
            MemorySystemConfig(
                cortex=CortexConfig(
                    input_dim=64, hidden_dims=(32,), latent_dim=8, epochs=5, batch_size=8
                ),
                seed=0,
            ),
            embedder=HashingEmbedder(dim=64, seed=0),
        )

    isolated = build()
    isolated.bootstrap(CORPUS)

    interleaved = build()
    _ = build()  # a second system in between
    interleaved.bootstrap(CORPUS)

    a = isolated.cortex.surprise(isolated.embedder.encode(CORPUS))
    b = interleaved.cortex.surprise(interleaved.embedder.encode(CORPUS))
    assert torch.allclose(a, b, atol=1e-6)


def test_config_is_not_mutated_in_place():
    config = MemorySystemConfig(
        cortex=CortexConfig(input_dim=999, hidden_dims=(32,), latent_dim=8, epochs=1)
    )
    OrganizationalMemory(config, embedder=HashingEmbedder(dim=64, seed=0))
    assert config.cortex.input_dim == 999


# ------------------------------------------------------------------- config guards


@pytest.mark.parametrize(
    "kwargs",
    [
        {"novelty": NoveltyConfig(quantile=1.5)},
        {"hopfield": HopfieldConfig(beta=0.0)},
        {"consolidation": ConsolidationConfig(relative_drop=1.5)},
        {"consolidation": ConsolidationConfig(replay_sigma=0.0)},
        {"consolidation": ConsolidationConfig(replay_sigma=-0.05)},
        {"consolidation": ConsolidationConfig(episodic_ratio=2.0)},
    ],
)
def test_invalid_config_raises_at_construction(kwargs):
    """These used to surface only mid-run — an out-of-range quantile raised from
    torch after training, and relative_drop >= 1 silently reintroduced an
    already-fixed defect (it pruned memories from an untouched cortex)."""
    with pytest.raises(ValueError):
        MemorySystemConfig(**kwargs)


# --------------------------------------------------------------------- lifecycle


def test_sleep_on_a_fresh_system_returns_an_empty_report():
    """torch.cat([]) raised before the guard that was meant to catch this."""
    system = make_system()
    system.bootstrap(CORPUS)
    report = system.sleep()
    assert report.replayed == 0
    assert report.new_items == 0


def test_gist_accepts_an_explicit_beta():
    system = make_system()
    system.bootstrap(CORPUS)
    system.log_event("the shard rebalancer corrupted the orders index")
    assert system.gist("shard rebalancer", beta=4.0).beta == pytest.approx(4.0)


def test_bootstrap_twice_does_not_stale_latent_keys():
    """bootstrap() trains the cortex, so latent-derived keys go stale exactly as
    they do after consolidate() — which already re-indexed, while bootstrap did
    not."""
    config = MemorySystemConfig(
        cortex=CortexConfig(
            input_dim=64, hidden_dims=(48,), latent_dim=16, epochs=60,
            batch_size=8, learning_rate=3e-3,
        ),
        novelty=NoveltyConfig(quantile=0.8, warmup=4, window=64),
        key=KeyConfig(mode=HippocampalKey.LATENT),
        seed=0,
    )
    system = OrganizationalMemory(config, embedder=HashingEmbedder(dim=64, seed=0))
    system.bootstrap(CORPUS)
    text = "the shard rebalancer corrupted the orders index during failover"
    record = system.log_event(text).record
    assert record is not None

    system.bootstrap(CORPUS)

    fresh = system.retrieval.encode_cue(text)
    cos = float(torch.nn.functional.cosine_similarity(record.key, fresh, dim=0))
    assert cos > 0.99, f"stored key went stale after re-bootstrap (cos={cos})"


# -------------------------------------------------------------------------- scale


def test_write_is_amortised_linear():
    """Reallocating the whole buffer per write made ingestion O(N^2): doubling N
    cost ~3.6x the time, hours of pure copying at 100k memories."""
    def elapsed(n: int) -> float:
        net = ModernHopfieldNetwork(256, HopfieldConfig())
        rows = torch.nn.functional.normalize(torch.randn(n, 256), dim=-1)
        start = time.perf_counter()
        for i in range(n):
            net.write(rows[i])
        return time.perf_counter() - start

    elapsed(500)  # warm up
    small, large = elapsed(1000), elapsed(4000)
    # 4x the writes should cost well under quadratic (16x); allow generous slack
    # for a noisy shared machine while still failing on true O(N^2).
    assert large < small * 10, f"write scaling looks quadratic: {small=} {large=}"


def test_capacity_growth_keeps_patterns_exact():
    """The growth buffer must never leak unused capacity into the public view."""
    net = ModernHopfieldNetwork(8, HopfieldConfig())
    for i in range(20):
        vec = torch.zeros(8)
        vec[i % 8] = 1.0
        net.write(vec)
        assert net.patterns.shape == (i + 1, 8)
        assert net.log_prior.shape == (i + 1,)
    assert len(net) == 20
    net.remove([0, 1, 2])
    assert net.patterns.shape == (17, 8)
    assert len(net) == 17
    net.set_log_prior(torch.zeros(17))
    assert net.attention(torch.eye(8)[0]).shape == (17,)


def test_lsa_fit_uses_a_sparse_matrix():
    """The dense fit allocated len(texts) x len(vocab) floats — 2.1 GB peak at
    10k documents, ~8 GB extrapolated at 100k."""
    corpus = [f"document number {i} about topic {i % 7} and other words" for i in range(400)]
    embedder = LatentSemanticEmbedder(dim=32, min_df=1, seed=0)
    sparse = embedder  # fit() must not raise or blow up
    sparse.fit(corpus)
    assert sparse.is_fitted
    vectors = sparse.encode(corpus[:5])
    assert vectors.shape == (5, 32)
    assert torch.isfinite(vectors).all()
