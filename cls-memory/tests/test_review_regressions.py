"""Regression guards for defects found in review.

Each test here corresponds to a specific bug that was verified numerically
before it was fixed. They also close the coverage gaps the review identified:
energy descent under the two extensions this module adds (non-uniform priors
and masking), the normalisation of `log_density`, and the *distribution*
produced by `langevin_replay` rather than merely its shape.
"""

from __future__ import annotations

import math

import pytest
import torch

from cls_memory import (
    CortexConfig,
    DecayConfig,
    HippocampalKey,
    HopfieldConfig,
    KeyConfig,
    MemoryRecord,
    MemoryStore,
    MemorySystemConfig,
    NoveltyConfig,
    OrganizationalMemory,
    Persistence,
    SlowLearningNeocortex,
)
from cls_memory.energy import basin_depth, langevin_replay, log_density, score
from cls_memory.hippocampus import ModernHopfieldNetwork
from tests.test_memory_system import CORPUS, make_system


# ------------------------------------------------------- S1: unnormalised keys


@pytest.mark.parametrize("normalize", [True, False])
def test_score_matches_log_density_gradient_at_any_pattern_scale(normalize):
    """The diffusion identities must hold in the configuration consolidation
    actually runs in (normalize_patterns=False), not only for unit vectors.

    Before the fix this diverged by ~24 in gradient norm and the Gibbs measure
    was a norm-tilted mixture rather than the memory distribution.
    """
    torch.manual_seed(0)
    net = ModernHopfieldNetwork(
        8, HopfieldConfig(beta=4.0, normalize_patterns=normalize)
    )
    net.write(torch.randn(5, 8) * 1.5)
    net.set_log_prior(torch.randn(5))

    xi = torch.randn(8, dtype=torch.float64, requires_grad=True)
    net.patterns = net.patterns.double()
    net.log_prior = net.log_prior.double()
    log_density(net, xi).backward()
    assert torch.allclose(score(net, xi.detach()), xi.grad, atol=1e-8)


@pytest.mark.parametrize("normalize", [True, False])
def test_gibbs_measure_is_proportional_to_the_mixture_at_any_scale(normalize):
    """log p + beta*E must be the same constant everywhere."""
    torch.manual_seed(0)
    net = ModernHopfieldNetwork(
        6, HopfieldConfig(beta=3.0, normalize_patterns=normalize)
    )
    net.write(torch.randn(4, 6) * 2.0)
    offsets = []
    for _ in range(6):
        z = torch.randn(6)
        offsets.append(float(log_density(net, z)) + 3.0 * float(net.energy(z)))
    assert max(offsets) - min(offsets) < 1e-3


def test_log_density_integrates_to_one():
    """Checks the normalising constant, which a peak-location test cannot."""
    torch.manual_seed(0)
    net = ModernHopfieldNetwork(2, HopfieldConfig(beta=4.0, normalize_patterns=False))
    net.write(torch.tensor([[0.5, 0.2], [-0.4, 0.6]]))
    net.set_log_prior(torch.tensor([math.log(0.3), math.log(0.7)]))

    lim, steps = 6.0, 601
    axis = torch.linspace(-lim, lim, steps)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
    density = log_density(net, grid).exp()
    cell = (2 * lim / (steps - 1)) ** 2
    assert float(density.sum() * cell) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------- energy descent under extensions


def test_energy_descends_with_a_non_uniform_prior():
    """CCCP descent must survive the log-prior extension -- untested before."""
    torch.manual_seed(0)
    for _ in range(20):
        net = ModernHopfieldNetwork(16, HopfieldConfig(beta=6.0))
        net.write(torch.randn(8, 16))
        net.set_log_prior(torch.randn(8) * 4.0)
        trace = net.retrieve(torch.randn(16) * 2.0, max_iter=12, tol=0.0)
        for a, b in zip(trace.energy_path, trace.energy_path[1:]):
            assert b <= a + 1e-5, f"energy increased {a} -> {b}"


def test_energy_descends_under_masking():
    """Clamping is constrained CCCP on an affine subspace; descent still holds."""
    torch.manual_seed(1)
    for _ in range(20):
        net = ModernHopfieldNetwork(32, HopfieldConfig(beta=8.0))
        net.write(torch.randn(10, 32))
        net.set_log_prior(torch.randn(10) * 2.0)
        mask = torch.rand(32) < 0.5
        trace = net.retrieve(torch.randn(32), mask=mask, max_iter=12, tol=0.0)
        for a, b in zip(trace.energy_path, trace.energy_path[1:]):
            assert b <= a + 1e-5, f"energy increased {a} -> {b}"


# ------------------------------------------------------------ S2: separation


def test_effective_separation_accounts_for_the_prior():
    """A geometrically well-separated but decayed memory is NOT a fixed point;
    raw `separation` cannot see that, `effective_separation` can."""
    torch.manual_seed(0)
    net = ModernHopfieldNetwork(32, HopfieldConfig(beta=8.0))
    net.write(torch.randn(4, 32))
    net.set_log_prior(torch.tensor([-25.0, 0.0, 0.0, 0.0]))

    assert net.separation(0) > 0.5  # looks healthy geometrically
    assert net.effective_separation(0) < 0  # but is not a fixed point
    assert int(net.retrieve(net.patterns[0].clone()).weights.argmax()) != 0


def test_separation_rejects_out_of_range_index():
    net = ModernHopfieldNetwork(8, HopfieldConfig())
    net.write(torch.randn(3, 8))
    with pytest.raises(IndexError):
        net.separation(-1)
    with pytest.raises(IndexError):
        net.separation(3)


# --------------------------------------------------------------- S3: basin


def test_basin_depth_is_scale_free_in_nats():
    """`depth` carries units of 1/beta; `depth_nats` must not."""
    torch.manual_seed(0)
    patterns = torch.randn(5, 16)
    query = torch.randn(16)
    nats = []
    for beta in (4.0, 16.0, 64.0):
        net = ModernHopfieldNetwork(16, HopfieldConfig(beta=beta))
        net.write(patterns)
        report = basin_depth(net, query)
        assert report.depth_nats == pytest.approx(beta * report.depth, rel=1e-5)
        nats.append(report.depth_nats)
    assert all(n > 0 for n in nats)


def test_basin_depth_rejects_batches_and_empty_memory():
    net = ModernHopfieldNetwork(8, HopfieldConfig())
    with pytest.raises(RuntimeError):
        basin_depth(net, torch.randn(8))
    net.write(torch.randn(3, 8))
    with pytest.raises(ValueError):
        basin_depth(net, torch.randn(2, 8))


def test_masked_cue_is_not_flagged_as_confabulation():
    """A masked cue has norm ~sqrt(keep), which used to depress top_similarity
    and halve confidence on a perfect completion."""
    system = make_system()
    system.bootstrap(CORPUS)
    text = "the shard rebalancer corrupted the orders index during failover"
    record = system.log_event(text).record
    system.log_event("a contractor deleted the staging kubernetes namespace")
    assert record is not None

    partial, mask = system.retrieval.occlude(text, keep_fraction=0.4, seed=0)
    result = system.complete(partial, mask, reinforce=False)
    assert result.top.record.id == record.id
    assert not result.basin.is_confabulation
    assert result.confidence > 0.9


# ------------------------------------------------------- S5: key invalidation


def test_consolidation_reindexes_latent_keys():
    """Training the cortex moves the encoder; stored latent keys must be
    re-encoded or the read path queries a space the store no longer uses.
    Measured cosine before the fix: 0.39 for the same text."""
    config = MemorySystemConfig(
        cortex=CortexConfig(
            input_dim=64, hidden_dims=(48,), latent_dim=16, epochs=200,
            batch_size=8, learning_rate=3e-3,
        ),
        novelty=NoveltyConfig(quantile=0.8, warmup=4, window=64),
        key=KeyConfig(mode=HippocampalKey.LATENT),
        seed=0,
    )
    system = OrganizationalMemory(config)
    system.bootstrap(CORPUS)
    text = "the shard rebalancer corrupted the orders index during failover"
    record = system.log_event(text).record
    assert record is not None

    system.consolidation.config.prune_predicted = False
    system.sleep(epochs=20)

    fresh = system.retrieval.encode_cue(text)
    cos = float(torch.nn.functional.cosine_similarity(record.key, fresh, dim=0))
    assert cos > 0.99, f"stored key drifted from the live encoder (cos={cos})"
    assert torch.allclose(system.store.mhn.patterns[0], record.key, atol=1e-5)


# ------------------------------------------------------------ S6: store/record


def test_record_key_does_not_alias_latent():
    record = MemoryRecord(text="x", embedding=torch.zeros(4), latent=torch.ones(4) / 2)
    record.latent[0] = 99.0
    assert float(record.key[0]) == 0.5


def test_store_keys_match_hopfield_patterns_for_unnormalised_input():
    store = MemoryStore(4)
    store.add(
        MemoryRecord(
            text="y", embedding=torch.zeros(4), latent=torch.tensor([3.0, 0.0, 0.0, 4.0])
        )
    )
    assert torch.allclose(store.keys()[0], store.mhn.patterns[0], atol=1e-6)
    assert float(store.keys()[0].norm()) == pytest.approx(1.0, abs=1e-6)


def test_store_rejects_duplicate_ids_and_bad_key_dims():
    store = MemoryStore(4)
    record = MemoryRecord(text="a", embedding=torch.zeros(4), latent=torch.ones(4))
    store.add(record)
    with pytest.raises(ValueError):
        store.add(record)
    with pytest.raises(ValueError):
        store.add(MemoryRecord(text="b", embedding=torch.zeros(8), latent=torch.ones(8)))


# ------------------------------------------------------------- S7: read path


def test_gist_does_not_reinforce():
    """A metastable mixture's top memory is an artefact of the blend, so
    bumping its strength and resetting its decay clock would be wrong."""
    system = make_system()
    system.bootstrap(CORPUS)
    for text in CORPUS[:4]:
        system.ingest(text, persistence=Persistence.EVERGREEN)
    system.gist("deployment service")
    assert all(r.access_count == 0 for r in system.records)


def test_basin_is_unaffected_by_reinforcement():
    """Reinforcement used to run between settling and the basin report, so the
    basin described post-reinforcement priors while the trace described
    pre-reinforcement ones. The reported basin must not depend on whether the
    caller asked for reinforcement."""
    query = "shard rebalancer orders index"

    def run(reinforce: bool) -> float:
        system = make_system()
        system.bootstrap(CORPUS)
        record = system.log_event(
            "the shard rebalancer corrupted the orders index"
        ).record
        assert record is not None
        record.age_by(90)
        return system.recall(query, reinforce=reinforce).basin.energy

    assert run(True) == pytest.approx(run(False), abs=1e-6)


# ---------------------------------------------------------------- S8: replay


def test_langevin_replay_samples_the_right_distribution():
    """One pattern, one dimension: the target is N(x, 1/beta) exactly."""
    beta = 4.0
    net = ModernHopfieldNetwork(1, HopfieldConfig(beta=beta, normalize_patterns=False))
    net.write(torch.tensor([[1.0]]))

    generator = torch.Generator().manual_seed(0)
    samples = langevin_replay(
        net, 40_000, steps=300, step_size=0.02, init_noise=1.0, generator=generator
    )
    assert float(samples.mean()) == pytest.approx(1.0, abs=0.02)
    assert float(samples.std()) == pytest.approx(1 / math.sqrt(beta), rel=0.05)


def test_replay_seeding_respects_salience():
    """Decayed memories should be replayed less; a too-wide seed noise used to
    wash this out into a uniform draw around the centroid."""
    torch.manual_seed(0)
    net = ModernHopfieldNetwork(32, HopfieldConfig(beta=32.0))
    net.write(torch.randn(4, 32))
    net.set_log_prior(torch.tensor([0.0, -6.0, -6.0, -6.0]))

    generator = torch.Generator().manual_seed(0)
    samples = langevin_replay(net, 400, steps=4, step_size=0.01, generator=generator)
    nearest = (samples @ net.patterns.T).argmax(dim=-1)
    assert int((nearest == 0).sum()) > 200, "salient memory was not replayed most"


# ----------------------------------------------------------------- S10: elbo


def test_elbo_mode_uses_full_kl_not_the_training_weight():
    cortex = SlowLearningNeocortex(
        CortexConfig(input_dim=8, hidden_dims=(8,), latent_dim=4, kl_weight=0.01)
    )
    x = torch.randn(3, 8)
    recon, kl = cortex.losses(x, sample=False)
    assert torch.allclose(cortex.surprise(x, "recon"), recon, atol=1e-5)
    assert torch.allclose(cortex.surprise(x, "elbo"), recon + kl, atol=1e-5)


def test_surprise_restores_training_mode():
    cortex = SlowLearningNeocortex(CortexConfig(input_dim=8, hidden_dims=(8,), latent_dim=4))
    cortex.train()
    cortex.surprise(torch.randn(2, 8))
    assert cortex.training
    cortex.latent(torch.randn(8))
    assert cortex.training


# ------------------------------------------------- S11: replay content / pruning


def _system_with_episodes(episodic_ratio: float) -> OrganizationalMemory:
    """A trained system holding five off-schema episodes."""
    torch.manual_seed(0)
    system = make_system()
    system.bootstrap(CORPUS)
    for text in (
        "the shard rebalancer corrupted the orders index during failover",
        "a contractor deleted the staging kubernetes namespace by accident",
        "the payment provider rotated credentials without any notice",
        "an unfamiliar vendor invoice arrived from a shell company in belize",
        "molten glass rained over the harbour as the volcano erupted",
    ):
        system.log_event(text)
    assert len(system) == 5
    system.config.consolidation.episodic_ratio = episodic_ratio
    return system


def _nearest_cosine(samples: torch.Tensor, stored: torch.Tensor) -> float:
    """Mean cosine from each replay sample to the closest stored embedding."""
    sim = (
        torch.nn.functional.normalize(samples, dim=-1)
        @ torch.nn.functional.normalize(stored, dim=-1).T
    )
    return float(sim.max(dim=-1).values.mean())


def test_replay_reinstates_the_stored_episodes():
    """Replay is supposed to reinstate hippocampal traces in the cortex. The
    decoder path does not: measured on the benchmark corpus, `decode(latent)`
    of a stored anomaly had cosine 0.047 with the embedding it came from,
    because the decoder was trained on routine text and has no resolution where
    novel episodes live. Only the embedding-space component carries the episode.
    """
    system = _system_with_episodes(0.5)
    stored = system.store.embeddings()

    episodic = system.consolidation._replay_episodic(64)
    generated = system.consolidation._replay_generated(64)

    assert _nearest_cosine(episodic, stored) > 0.8, "episodic replay lost the episodes"
    assert _nearest_cosine(generated, stored) < 0.6, (
        "the generated component unexpectedly reinstates episodes; if this ever "
        "becomes true the episodic component may no longer be needed"
    )
    assert _nearest_cosine(system.consolidation.replay(64), stored) > 0.6


def test_replay_is_numerically_stable_at_the_configured_noise_level():
    """Guard on the ULA step-size cap in `_replay_episodic`.

    The embedding landscape runs at beta = 1/replay_sigma^2 = 400, where the
    unadjusted Langevin map contracts by (1 - beta*eta/2) per step and diverges
    for eta > 4/beta = 0.01. The configured `replay_step_size` of 0.05 is five
    times that, so using it unclamped multiplies the state by -9 every step.
    """
    system = _system_with_episodes(1.0)
    stored = system.store.embeddings()
    samples = system.consolidation.replay(64)
    assert torch.isfinite(samples).all()
    ratio = float(samples.norm(dim=-1).mean() / stored.norm(dim=-1).mean())
    assert 0.5 < ratio < 2.0, f"replay samples left the embedding scale (x{ratio:.2f})"


def test_decoder_only_replay_teaches_the_cortex_nothing():
    """The defect this section exists for.

    Training the VAE on its own decoder output is self-distillation: the target
    is already what the model emits. Measured on the benchmark, surprise on the
    stored memories after a 150-epoch pass was 1.0016x its value at ingestion,
    so `prune_predicted` was correct and permanently inert -- 0 of 36 memories
    released. The comparison against the fixed default is the point: this is
    not a claim that pruning is hard, it is a claim that replay was empty.
    """
    naive = _system_with_episodes(0.0)
    naive.consolidation.config.prune_predicted = False
    naive.consolidation.consolidate(None, epochs=60)
    naive_ratios = list(naive.consolidation.drop_ratios().values())

    assert min(naive_ratios) > 0.9, (
        "decoder-only replay moved surprise; the self-distillation reading of "
        f"the bug no longer holds (min drop ratio {min(naive_ratios):.4f})"
    )
    assert naive.consolidation.prune_predicted() == []

    fixed = _system_with_episodes(0.5)
    fixed.consolidation.config.prune_predicted = False
    fixed.consolidation.consolidate(None, epochs=60)
    fixed_ratios = list(fixed.consolidation.drop_ratios().values())

    assert min(fixed_ratios) < 0.5 * min(naive_ratios)
    assert len(fixed.consolidation.prune_predicted()) >= 3


def test_generated_replay_share_is_what_protects_the_schema():
    """Why `episodic_ratio` is 0.125 and not 1.0.

    Replaying only the stored episodes learns them fastest and takes the schema
    down doing it. Measured on the full benchmark, replay's drift reduction
    against naive training falls 55.3% -> 54.9% -> 50.3% -> 46.0% as the ratio
    goes 0 -> 0.125 -> 0.25 -> 0.5, and on this corpus a replay-only pass at
    ratio 1.0 drifts the schema roughly twice as far as at 0.5. Pruning bought
    at that price is not worth having, so the ordering is asserted rather than
    just the pruning.
    """
    old = None
    drift = {}
    for ratio in (0.5, 1.0):
        system = _system_with_episodes(ratio)
        if old is None:
            old = system.embedder.encode(CORPUS)
        before = float(system.cortex.surprise(old).mean())
        system.consolidation.config.prune_predicted = False
        system.consolidation.consolidate(None, epochs=60)
        drift[ratio] = float(system.cortex.surprise(old).mean()) - before

    assert drift[1.0] > drift[0.5], (
        "pure episodic replay no longer damages the schema more than the "
        f"mixture ({drift[1.0]:+.4f} vs {drift[0.5]:+.4f}); re-derive the default"
    )
