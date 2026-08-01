"""Tests for the Modern Hopfield Network against the properties Ramsauer et al.
prove: energy descent, one-step convergence, exponentially small retrieval
error for well-separated patterns, and metastable mixtures at low beta.
"""

from __future__ import annotations

import math

import pytest
import torch

from cls_memory.config import HopfieldConfig
from cls_memory.energy import beta_to_sigma, denoise, log_density, score
from cls_memory.hippocampus import ModernHopfieldNetwork


def make_net(n=8, d=16, beta=8.0, seed=0):
    torch.manual_seed(seed)
    net = ModernHopfieldNetwork(d, HopfieldConfig(beta=beta))
    x = torch.randn(n, d)
    net.write(x)
    return net


def test_write_normalizes_and_indexes():
    net = make_net(n=5, d=8)
    assert len(net) == 5
    assert torch.allclose(net.patterns.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_energy_is_monotonically_non_increasing():
    """CCCP guarantees the update never raises the energy."""
    net = make_net(n=12, d=32, beta=4.0)
    xi = torch.randn(32)
    xi = xi / xi.norm()
    trace = net.retrieve(xi, max_iter=16)
    path = trace.energy_path
    for a, b in zip(path, path[1:]):
        assert b <= a + 1e-5, f"energy increased: {a} -> {b}"


def test_stored_pattern_is_a_fixed_point():
    """With high beta and random (well-separated) patterns, each stored pattern
    retrieves itself with exponentially small error."""
    net = make_net(n=10, d=64, beta=32.0)
    for i in range(len(net)):
        trace = net.retrieve(net.patterns[i].clone())
        assert int(trace.weights.argmax()) == i
        err = float((trace.state - net.patterns[i]).norm())
        assert err < 1e-2, f"pattern {i} not a fixed point (err={err})"


def test_converges_in_one_step_from_a_clean_cue():
    net = make_net(n=10, d=64, beta=32.0)
    trace = net.retrieve(net.patterns[3].clone(), max_iter=8)
    assert trace.converged
    assert trace.iterations <= 2


def test_noisy_cue_recovers_the_right_pattern():
    net = make_net(n=10, d=64, beta=32.0)
    torch.manual_seed(1)
    target = net.patterns[7]
    noisy = target + 0.35 * torch.randn(64)
    trace = net.retrieve(noisy / noisy.norm())
    assert int(trace.weights.argmax()) == 7
    assert not trace.is_mixture


def test_low_beta_produces_a_metastable_mixture():
    """Low beta = high equivalent noise = broad basins spanning many memories."""
    net = make_net(n=16, d=32, beta=0.05)
    xi = net.patterns[0].clone()
    trace = net.retrieve(xi)
    assert trace.is_mixture
    assert float(trace.weights.max()) < 0.5


def test_masked_retrieval_clamps_known_coordinates():
    net = make_net(n=6, d=32, beta=16.0)
    target = net.patterns[2]
    mask = torch.zeros(32, dtype=torch.bool)
    mask[:16] = True
    cue = target * mask
    trace = net.retrieve(cue, mask=mask)
    assert torch.allclose(trace.state[mask], cue[mask], atol=1e-6)
    assert int(trace.weights.argmax()) == 2


def test_partial_cue_completes_the_missing_half():
    """Pattern completion proper: only 40% of coordinates are given."""
    net = make_net(n=8, d=64, beta=32.0)
    torch.manual_seed(3)
    target = net.patterns[5]
    mask = torch.rand(64) < 0.4
    cue = target * mask
    trace = net.retrieve(cue, mask=mask, max_iter=32)
    completed = trace.state[~mask]
    truth = target[~mask]
    cos = float(torch.nn.functional.cosine_similarity(completed, truth, dim=0))
    assert cos > 0.8, f"completion cosine only {cos}"


def test_log_prior_biases_retrieval():
    """Salience must be able to decide between two equally similar memories."""
    torch.manual_seed(0)
    net = ModernHopfieldNetwork(32, HopfieldConfig(beta=8.0))
    a = torch.randn(32)
    b = a + 0.25 * torch.randn(32)  # deliberately close to a
    net.write(torch.stack([a, b]))

    query = (net.patterns[0] + net.patterns[1]) / 2
    net.set_log_prior(torch.tensor([math.log(1.0), math.log(0.01)]))
    assert int(net.attention(query).argmax()) == 0
    net.set_log_prior(torch.tensor([math.log(0.01), math.log(1.0)]))
    assert int(net.attention(query).argmax()) == 1


def test_uniform_prior_matches_ramsauer_energy():
    """With w_i = 1/N our energy must equal the paper's eq. 4 exactly."""
    net = make_net(n=9, d=16, beta=3.0)
    xi = torch.randn(16)
    beta, n = 3.0, len(net)
    lse = torch.logsumexp(beta * (net.patterns @ xi), dim=0) / beta
    max_sq = float((net.patterns.norm(dim=-1) ** 2).max())
    expected = -lse + 0.5 * xi @ xi + math.log(n) / beta + 0.5 * max_sq
    assert float(net.energy(xi)) == pytest.approx(float(expected), abs=1e-5)


def test_analytic_gradient_matches_autograd():
    net = make_net(n=7, d=16, beta=5.0)
    xi = torch.randn(16, requires_grad=True)
    net.energy(xi).backward()
    analytic = net.grad_energy(xi.detach())
    assert torch.allclose(analytic, xi.grad, atol=1e-5)


def test_remove_compacts_patterns_and_priors():
    net = make_net(n=6, d=8)
    kept = net.patterns[[0, 2, 4]].clone()
    net.remove([1, 3, 5])
    assert len(net) == 3
    assert net.log_prior.shape[0] == 3
    assert torch.allclose(net.patterns, kept)


def test_separation_flags_near_duplicates():
    torch.manual_seed(0)
    net = ModernHopfieldNetwork(32, HopfieldConfig(beta=8.0))
    a = torch.randn(32)
    near = a + 0.02 * torch.randn(32)
    far = torch.randn(32)
    net.write(torch.stack([a, near, far]))
    assert net.separation(0) < net.separation(2)


def test_empty_network_rejects_retrieval():
    net = ModernHopfieldNetwork(8)
    assert net.is_empty
    with pytest.raises(RuntimeError):
        net.retrieve(torch.randn(8))



# ------------------------------------------------------- diffusion equivalence


def test_update_equals_tweedie_denoiser():
    """One MHN step == posterior mean E[x|xi] == one diffusion denoising step."""
    net = make_net(n=10, d=32, beta=6.0)
    xi = torch.randn(32)
    sigma_sq = beta_to_sigma(6.0) ** 2
    tweedie = xi + sigma_sq * score(net, xi)
    assert torch.allclose(denoise(net, xi), tweedie, atol=1e-5)
    assert torch.allclose(net.step(xi), tweedie, atol=1e-5)


def test_score_equals_negative_beta_times_energy_gradient():
    net = make_net(n=10, d=32, beta=6.0)
    xi = torch.randn(32, requires_grad=True)
    net.energy(xi).backward()
    assert torch.allclose(score(net, xi.detach()), -6.0 * xi.grad, atol=1e-4)


def test_gibbs_measure_is_the_gaussian_mixture():
    """exp(-beta E) must be proportional to the Gaussian mixture density, i.e.
    log p + beta*E is the same constant at every point in space."""
    net = make_net(n=8, d=16, beta=4.0)
    torch.manual_seed(11)
    offsets = []
    for _ in range(6):
        xi = torch.randn(16)
        offsets.append(float(log_density(net, xi) + 4.0 * float(net.energy(xi))))
    assert max(offsets) - min(offsets) < 1e-3


def test_log_density_peaks_on_stored_patterns():
    net = make_net(n=8, d=16, beta=16.0)
    on = float(log_density(net, net.patterns[0]))
    off = float(log_density(net, torch.randn(16) / 4))
    assert on > off
