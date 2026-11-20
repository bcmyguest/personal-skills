"""Tests for the pure, LM-free pieces of ticket 13's learned read-in
(`experiments/kv_learned_readin.py`): `LearnedReadIn`'s shape/geometry
contract, that `train_readin` actually reduces a loss by gradient descent,
and that `kv_injection.noise_like` -- the matched-noise control this module
reuses for the `kv_learned_all` arm -- is actually matched in norm/variance
to the signal it is a floor for.

Deliberately does NOT touch SmolLM2 or any other decoder LM: no download, no
multi-minute training run. `differentiable_nll` / `recitation_loss` (which
need the real model) are exercised by `kv_learned_readin.py`'s own smoke
`main()`, not here.
"""

from __future__ import annotations

import torch

from experiments.kv_injection import KV, noise_like
from experiments.kv_learned_readin import LearnedReadIn, detach_kv, train_readin

N_LAYERS = 3
KV_HEADS = 2
HEAD_DIM = 4


def make_kv(seed: int = 0, slots: int = 1) -> KV:
    """A tiny, fixture-scale KV shaped like a real one-slot superposition:
    (1, kv_heads, slots, head_dim) per layer, for keys and values."""
    g = torch.Generator().manual_seed(seed)
    shape = (1, KV_HEADS, slots, HEAD_DIM)
    keys = tuple(torch.randn(shape, generator=g) for _ in range(N_LAYERS))
    values = tuple(torch.randn(shape, generator=g) for _ in range(N_LAYERS))
    return KV(keys, values)


# --------------------------------------------------------- shape / geometry


def test_forward_preserves_shape():
    """Output shape must equal input shape per layer -- the projection moves
    content, it does not change how many slots/heads/dims the model sees."""
    readin = LearnedReadIn(N_LAYERS, HEAD_DIM)
    kv = make_kv()
    out = readin(kv)
    assert len(out.keys) == len(kv.keys) == N_LAYERS
    for i in range(N_LAYERS):
        assert out.keys[i].shape == kv.keys[i].shape
        assert out.values[i].shape == kv.values[i].shape


def test_forward_rejects_layer_count_mismatch():
    """A KV built for the wrong number of layers must fail loudly, not
    silently broadcast or truncate -- the same defensive posture
    `ModernHopfieldNetwork.write` takes for malformed input."""
    readin = LearnedReadIn(N_LAYERS, HEAD_DIM)
    wrong = make_kv()
    short = KV(wrong.keys[:-1], wrong.values[:-1])
    try:
        readin(short)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a layer-count mismatch")


def test_identity_init_is_a_no_op():
    """`eye_` weight + zero bias means step-0 training starts from Part VII's
    untrained baseline (plain `superpose`), not from noise: forward(kv) must
    equal kv itself before any gradient step touches the parameters."""
    readin = LearnedReadIn(N_LAYERS, HEAD_DIM)
    kv = make_kv(seed=1)
    out = readin(kv)
    for i in range(N_LAYERS):
        assert torch.allclose(out.keys[i], kv.keys[i], atol=1e-6)
        assert torch.allclose(out.values[i], kv.values[i], atol=1e-6)


# --------------------------------------------------------------- training


def test_train_readin_reduces_loss():
    """A synthetic, LM-free loss: MSE between readin(base) and a fixed random
    target. If gradient descent through `LearnedReadIn`'s own Linear layers
    works at all, this loss must fall substantially over enough Adam steps --
    the same property the real recitation loss needs in `main()`, minus the
    135M-parameter decoder."""
    torch.manual_seed(0)
    readin = LearnedReadIn(N_LAYERS, HEAD_DIM)
    base = make_kv(seed=2)
    target = make_kv(seed=3)

    def loss_fn(net: LearnedReadIn) -> torch.Tensor:
        out = net(base)
        total = torch.zeros(())
        for i in range(N_LAYERS):
            total = total + torch.nn.functional.mse_loss(out.keys[i], target.keys[i])
            total = total + torch.nn.functional.mse_loss(out.values[i], target.values[i])
        return total / N_LAYERS

    history = train_readin(readin, loss_fn, steps=50, lr=0.2)
    assert len(history) == 50
    # Loose bound: not asserting convergence to zero, just that training
    # actually moved the loss a lot, not by rounding-error noise.
    assert history[-1] < history[0] * 0.1, (history[0], history[-1])


def test_train_readin_rejects_zero_steps():
    readin = LearnedReadIn(N_LAYERS, HEAD_DIM)
    try:
        train_readin(readin, lambda net: torch.zeros(()), steps=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for steps=0")


def test_detach_kv_removes_grad():
    """`main()` hands `score_arm` a detached KV so scoring never builds a
    graph through the trained projection; verify the helper actually
    detaches every tensor, not just the first."""
    readin = LearnedReadIn(N_LAYERS, HEAD_DIM)
    kv = make_kv(seed=4)
    out = readin(kv)
    assert out.keys[0].requires_grad
    detached = detach_kv(out)
    for k, v in zip(detached.keys, detached.values):
        assert not k.requires_grad
        assert not v.requires_grad


# ------------------------------------------------------- matched-noise control


def test_noise_like_matches_reference_scale():
    """`noise_like` (reused unmodified from kv_injection.py, not
    reimplemented) is the floor control for `kv_learned_all`: same shape,
    same per-layer/per-tensor std as the signal it stands in for, or the
    comparison in RESULTS-PART-VII.md-style tables (real memory vs matched
    noise) is not actually matched."""
    reference = make_kv(seed=5, slots=3)
    noise = noise_like(reference)
    assert len(noise.keys) == len(reference.keys)
    for i in range(N_LAYERS):
        assert noise.keys[i].shape == reference.keys[i].shape
        assert noise.values[i].shape == reference.values[i].shape
        # matched per-layer scale (std), within sampling tolerance -- this is
        # exactly what noise_like promises: "same slot count, same norms, no
        # content" (kv_injection.py's own docstring for this function).
        ref_std = float(reference.keys[i].std())
        noise_std = float(noise.keys[i].std())
        assert abs(noise_std - ref_std) < 0.35 * ref_std, (i, ref_std, noise_std)
        # and it must not just be a copy of the reference -- "no content".
        assert not torch.allclose(noise.keys[i], reference.keys[i])


def test_noise_like_is_reproducible():
    """kv_injection.noise_like seeds its own generator (SEED), so two calls
    on the same reference must produce identical noise -- this is what makes
    the matched-noise numbers in RESULTS-PART-VII.md reproducible at all."""
    reference = make_kv(seed=6)
    a = noise_like(reference)
    b = noise_like(reference)
    for i in range(N_LAYERS):
        assert torch.equal(a.keys[i], b.keys[i])
        assert torch.equal(a.values[i], b.values[i])
