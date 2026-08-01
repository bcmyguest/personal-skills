"""Tests for decay, ingestion routing, retrieval and consolidation."""

from __future__ import annotations

import math
from datetime import timedelta

import pytest
import torch

from cls_memory import (
    ConsolidationConfig,
    CortexConfig,
    DecayConfig,
    HopfieldConfig,
    IngestionAction,
    MemoryRecord,
    MemorySystemConfig,
    NoveltyConfig,
    OrganizationalMemory,
    Persistence,
)
from cls_memory.records import utcnow

CORPUS = [
    "deployment of the billing service completed without incident",
    "deployment of the checkout service completed without incident",
    "deployment of the search service completed without incident",
    "nightly batch job finished successfully in the eu-west region",
    "nightly batch job finished successfully in the us-east region",
    "weekly report generated and delivered to the finance team",
    "weekly report generated and delivered to the operations team",
    "customer support ticket resolved within the standard response window",
    "customer support ticket closed within the standard response window",
    "routine database backup completed for the primary cluster",
    "routine database backup completed for the replica cluster",
    "scheduled maintenance window opened for the staging environment",
]


def make_system(**overrides) -> OrganizationalMemory:
    config = MemorySystemConfig(
        cortex=CortexConfig(
            input_dim=64,
            hidden_dims=(48,),
            latent_dim=16,
            epochs=300,
            batch_size=8,
            learning_rate=3e-3
        ),
        novelty=NoveltyConfig(quantile=0.8, warmup=4, window=64),
        hopfield=HopfieldConfig(beta=16.0),
        decay=DecayConfig(half_life_days=30.0),
        consolidation=ConsolidationConfig(replay_batch=16, replay_steps=4),
        seed=0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return OrganizationalMemory(config)


@pytest.fixture(scope="module")
def trained() -> OrganizationalMemory:
    system = make_system()
    system.bootstrap(CORPUS)
    return system


# ------------------------------------------------------------ evergreen/decay


def _record(persistence: Persistence) -> MemoryRecord:
    return MemoryRecord(
        text="x",
        embedding=torch.zeros(4),
        latent=torch.ones(4) / 2,
        persistence=persistence,
    )


def test_evergreen_never_decays():
    r = _record(Persistence.EVERGREEN)
    r.age_by(3650)
    assert r.salience(half_life_days=30.0) == 1.0


def test_temporal_halves_every_half_life():
    r = _record(Persistence.TEMPORAL)
    r.age_by(30)
    assert r.salience(half_life_days=30.0) == pytest.approx(0.5, abs=1e-6)
    r.age_by(30)
    assert r.salience(half_life_days=30.0) == pytest.approx(0.25, abs=1e-6)
    r.age_by(60)
    assert r.salience(half_life_days=30.0) == pytest.approx(0.0625, abs=1e-6)


def test_decay_is_exponential_in_age():
    r = _record(Persistence.TEMPORAL)
    r.age_by(45)
    expected = math.pow(2.0, -45 / 30)
    assert r.salience(half_life_days=30.0) == pytest.approx(expected, abs=1e-6)


def test_reinforcement_resets_the_clock_and_raises_strength():
    r = _record(Persistence.TEMPORAL)
    r.age_by(60)
    assert r.salience(half_life_days=30.0) < 0.3
    r.reinforce(gain=0.25)
    assert r.salience(half_life_days=30.0) == pytest.approx(1.25, abs=1e-6)
    assert r.access_count == 1


def test_sweep_prunes_decayed_but_keeps_evergreen(trained):
    system = make_system()
    system.bootstrap(CORPUS)
    system.remember_rule("all refunds above 500 dollars require director approval")
    system.log_event("pager alert fired for the payments gateway at 3am")

    for record in system.records:
        record.age_by(400)  # ~13 half-lives

    report = system.sweep()
    assert report.pruned_decayed >= 1
    kinds = {r.persistence for r in system.records}
    assert Persistence.TEMPORAL not in kinds
    assert Persistence.EVERGREEN in kinds


def test_decayed_memory_loses_to_a_fresh_one(trained):
    """The forgetting curve must actually change retrieval order, not just a
    number on the record."""
    system = make_system()
    system.bootstrap(CORPUS)
    old = system.log_event("the payments gateway timed out during checkout").record
    new = system.log_event("the payments gateway timed out during checkout again").record
    assert old is not None and new is not None

    old.age_by(365)
    system.store.refresh_priors()
    result = system.recall("payments gateway timeout", reinforce=False)
    assert result.top is not None
    assert result.top.record.id == new.id


def test_evergreen_outranks_an_equally_similar_stale_episode():
    system = make_system()
    system.bootstrap(CORPUS)
    rule = system.remember_rule("refunds above 500 dollars require director approval").record
    episode = system.log_event("refunds above 500 dollars were approved by a director").record
    assert rule is not None and episode is not None

    episode.age_by(300)
    system.store.refresh_priors()
    result = system.recall("refunds above 500 dollars director", reinforce=False)
    assert result.top is not None
    assert result.top.record.id == rule.id


# ------------------------------------------------------------------ ingestion


def test_novel_item_is_stored(trained):
    system = make_system()
    system.bootstrap(CORPUS)
    before = len(system)
    result = system.log_event(
        "catastrophic data loss in the shard rebalancer wiped 40 percent of orders"
    )
    assert result.action is IngestionAction.STORED
    assert result.novelty > result.threshold
    assert len(system) == before + 1


def test_best_predicted_corpus_item_is_pruned_not_stored():
    """The item the schema reconstructs best must never reach the hippocampus.

    Picked by argmin rather than hard-coded: a quantile-calibrated gate admits
    (1 - q) of its own calibration distribution *by construction*, so asserting
    that an arbitrary training sentence is PREDICTED is a coin flip on a corpus
    this small. The contract that actually holds is about the best-predicted
    item, and about the rate (next test).
    """
    system = make_system()
    system.bootstrap(CORPUS)
    scores = system.cortex.surprise(system.embedder.encode(CORPUS))
    easiest = CORPUS[int(scores.argmin())]

    before = len(system)
    result = system.log_event(easiest)
    assert result.action is IngestionAction.PREDICTED
    assert result.novelty <= result.threshold
    assert result.record is None
    assert len(system) == before


def test_most_of_the_training_corpus_is_pruned_as_predicted():
    """Routing rate, which is the property that matters operationally: the
    hippocampus must not fill up with things the schema already knows."""
    system = make_system()
    system.bootstrap(CORPUS)
    actions = [system.log_event(t).action for t in CORPUS]
    predicted = sum(a is IngestionAction.PREDICTED for a in actions)
    assert predicted >= 0.6 * len(CORPUS), f"only {predicted}/{len(CORPUS)} pruned"


def test_off_schema_items_are_all_stored():
    system = make_system()
    system.bootstrap(CORPUS)
    off_schema = [
        "the volcano erupted over the harbour and molten glass rained on the boats",
        "an unfamiliar vendor invoice arrived from a shell company in belize",
        "a contractor deleted the staging kubernetes namespace by hand",
    ]
    for text in off_schema:
        assert system.log_event(text).was_stored, text


def test_evergreen_bypasses_the_novelty_gate():
    system = make_system()
    system.bootstrap(CORPUS)
    result = system.remember_rule(
        "deployment of the billing service completed without incident"
    )
    assert result.was_stored
    assert result.record is not None
    assert result.record.is_evergreen


def test_duplicate_reinforces_instead_of_duplicating():
    system = make_system()
    system.bootstrap(CORPUS)
    first = system.log_event("the shard rebalancer corrupted the orders index")
    assert first.record is not None
    before = len(system)

    second = system.log_event("the shard rebalancer corrupted the orders index")
    assert second.action is IngestionAction.REINFORCED
    assert second.duplicate_of == first.record.id
    assert len(system) == before
    assert first.record.access_count == 1


def test_gate_calibration_tracks_the_corpus():
    system = make_system()
    system.bootstrap(CORPUS)
    assert system.gate.observed == len(CORPUS)
    assert math.isfinite(system.gate.threshold)


def test_novelty_ordering_is_sane(trained):
    """A wildly off-schema sentence must be more surprising than an on-schema one."""
    on = trained.embedder.encode(
        ["deployment of the payments service completed without incident"]
    )
    off = trained.embedder.encode(
        ["the volcano erupted over the harbour and molten glass rained on the boats"]
    )
    assert float(trained.cortex.surprise(off)) > float(trained.cortex.surprise(on))


# ------------------------------------------------------------------ retrieval


def test_recall_returns_the_matching_memory():
    system = make_system()
    system.bootstrap(CORPUS)
    stored = system.log_event(
        "the shard rebalancer corrupted the orders index during failover"
    ).record
    assert stored is not None
    system.log_event("a contractor deleted the staging kubernetes namespace")

    result = system.recall("shard rebalancer orders index", reinforce=False)
    assert result.top is not None
    assert result.top.record.id == stored.id


def test_partial_cue_completes_to_the_full_memory():
    system = make_system()
    system.bootstrap(CORPUS)
    text = "the shard rebalancer corrupted the orders index during failover"
    stored = system.log_event(text).record
    system.log_event("a contractor deleted the staging kubernetes namespace")
    system.log_event("the payment provider rotated credentials without notice")
    assert stored is not None

    partial, mask = system.retrieval.occlude(text, keep_fraction=0.4, seed=2)
    result = system.complete(partial, mask, reinforce=False)
    assert result.top is not None
    assert result.top.record.id == stored.id
    # Known coordinates must be preserved exactly.
    assert torch.allclose(result.trace.state[mask], partial[mask], atol=1e-6)


def test_recall_reinforces_the_winning_memory():
    system = make_system()
    system.bootstrap(CORPUS)
    record = system.log_event("the shard rebalancer corrupted the orders index").record
    assert record is not None
    record.age_by(45)
    system.recall("shard rebalancer orders", reinforce=True)
    assert record.access_count == 1
    assert record.age_days() < 1.0


def test_low_beta_recall_returns_a_gist():
    system = make_system()
    system.bootstrap(CORPUS)
    for text in CORPUS[:6]:
        system.ingestion.store.add(
            MemoryRecord(
                text=text,
                embedding=system.embedder.encode([text])[0],
                latent=system.retrieval.encode_cue(text),
            )
        )
    result = system.gist("deployment completed", reinforce=False)
    assert result.is_gist
    assert result.top is not None and result.top.weight < 0.5


def test_basin_depth_flags_an_out_of_distribution_query():
    system = make_system()
    system.bootstrap(CORPUS)
    system.log_event("the shard rebalancer corrupted the orders index")
    near = system.recall("shard rebalancer orders index", reinforce=False)
    far = system.recall(
        "quantum chromodynamics lattice gauge simulation", reinforce=False
    )
    assert far.basin.depth > near.basin.depth
    assert far.confidence <= near.confidence


def test_recall_on_empty_memory_raises():
    system = make_system()
    system.bootstrap(CORPUS)
    with pytest.raises(RuntimeError):
        system.recall("anything")


# -------------------------------------------------------------- consolidation


def test_replay_produces_embedding_shaped_samples():
    system = make_system()
    system.bootstrap(CORPUS)
    system.log_event("the shard rebalancer corrupted the orders index")
    system.log_event("a contractor deleted the staging kubernetes namespace")

    replayed = system.consolidation.replay(12)
    assert replayed.shape == (12, system.cortex.config.input_dim)
    assert torch.isfinite(replayed).all()


def test_consolidation_lowers_loss_on_its_own_batch():
    system = make_system()
    system.bootstrap(CORPUS)
    system.log_event("the shard rebalancer corrupted the orders index")
    report = system.sleep(["an unfamiliar vendor invoice arrived from a shell company"], epochs=15)
    assert report.replayed > 0
    assert report.improved


def test_consolidation_prunes_what_the_cortex_learns():
    """The whole point of the loop: once the schema absorbs an episode, the
    hippocampal trace should be released."""
    system = make_system()
    system.bootstrap(CORPUS)
    novel = "the shard rebalancer corrupted the orders index during failover"
    stored = system.log_event(novel).record
    assert stored is not None

    # Overtrain the cortex on that exact item so it becomes highly predictable.
    x = system.embedder.encode([novel] * 32)
    system.cortex.fit(x, epochs=200, lr=3e-3)

    pruned = system.consolidation.prune_predicted(threshold=float("inf"))
    assert stored.id in pruned
    assert len(system) == 0


def test_consolidation_protects_evergreen_records():
    system = make_system()
    system.bootstrap(CORPUS)
    rule = system.remember_rule("all refunds above 500 dollars need director approval").record
    episode = system.log_event("a refund of 800 dollars was approved on tuesday").record
    assert rule is not None and episode is not None

    pruned = system.consolidation.prune_predicted(threshold=float("inf"))
    assert rule.id not in pruned
    assert episode.id in pruned


def test_interleaved_replay_resists_catastrophic_forgetting():
    """Training only on new off-schema data should damage the old schema more
    than training on the same data interleaved with replay."""
    torch.manual_seed(0)
    intruder = [
        "molten glass rained over the harbour as the volcano erupted",
        "the volcano erupted again and ash covered every boat in the harbour",
    ] * 8

    baseline = make_system()
    baseline.bootstrap(CORPUS)
    old = baseline.embedder.encode(CORPUS)
    before = float(baseline.cortex.surprise(old).mean())

    # No replay: pure new-data training.
    naive = make_system()
    naive.bootstrap(CORPUS)
    for text in CORPUS[:6]:
        naive.log_event(text + " and then the volcano erupted")
    naive.cortex.fit(naive.embedder.encode(intruder), epochs=60, lr=1e-3)
    naive_drift = float(naive.cortex.surprise(old).mean()) - before

    # With replay interleaved.
    cls = make_system()
    cls.bootstrap(CORPUS)
    for text in CORPUS[:6]:
        cls.log_event(text + " and then the volcano erupted")
    cls.consolidation.config.prune_predicted = False
    cls.consolidation.consolidate(cls.embedder.encode(intruder), epochs=60)
    cls_drift = float(cls.cortex.surprise(old).mean()) - before

    assert cls_drift < naive_drift


# --------------------------------------------------------------------- store


def test_store_remove_keeps_records_and_patterns_aligned():
    system = make_system()
    system.bootstrap(CORPUS)
    texts = [
        "the shard rebalancer corrupted the orders index",
        "a contractor deleted the staging kubernetes namespace",
        "the payment provider rotated credentials without notice",
    ]
    records = [system.log_event(t).record for t in texts]
    assert all(r is not None for r in records), "setup: every event should store"
    before = len(system)

    system.store.remove([records[1].id])
    assert len(system) == before - 1
    for row, record in enumerate(system.store.records):
        assert system.store.row_of(record.id) == row
        assert torch.allclose(system.store.mhn.patterns[row], record.key, atol=1e-5)


def test_stats_reports_the_two_populations():
    system = make_system()
    system.bootstrap(CORPUS)
    system.remember_rule("invoices are paid net 30")
    system.log_event("an unfamiliar vendor invoice arrived from a shell company")
    stats = system.stats()
    assert stats["evergreen"] == 1
    assert stats["temporal"] >= 1
    assert stats["total"] == len(system)


def test_full_lifecycle_runs_end_to_end():
    system = make_system()
    system.bootstrap(CORPUS)
    system.remember_rule("production deploys are frozen during the december code freeze")
    system.log_event("someone deployed to production during the december code freeze")
    system.log_event("the incident review found no rollback plan existed")

    result = system.recall("december code freeze deploy", reinforce=True)
    assert result.top is not None

    for record in system.store.by_persistence(Persistence.TEMPORAL):
        record.age_by(200)
    report = system.sleep(epochs=5)
    assert report.sweep is not None

    assert any(r.is_evergreen for r in system.records)
