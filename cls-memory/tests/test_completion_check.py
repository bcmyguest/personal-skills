"""Guards for ticket 10 -- pattern completion measured against a real baseline.

VIII.2 retracts RESULTS.md section 4's *interpretation* on the strength of one
number section 4 never had: the single-shot cosine arm, scored on the identical
degraded cue. That retraction is only as trustworthy as two things -- the
occlusion that builds the cue, and the cosine arm that scores it -- so those are
what this file pins.

The failure mode worth guarding is not a crash. It is an occlusion that quietly
degrades the cue *less* than it claims, which would make both arms look good and
would have kept section 4's original claim alive for the same reason it survived
the first time: nothing measured alongside it.
"""

from __future__ import annotations

import torch

from experiments.completion_check import (
    KEEPS,
    _occlude_active_units,
    _occlude_coordinates,
)


def _sparse_key(n: int = 200, active: int = 40, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    key = torch.zeros(n)
    idx = torch.randperm(n, generator=g)[:active]
    key[idx] = torch.rand(active, generator=g) + 0.5
    return key


def test_coordinate_occlusion_keeps_about_the_requested_fraction():
    key = _sparse_key(n=4000, active=800)
    for keep in (0.8, 0.5, 0.1):
        mask = _occlude_coordinates(key, keep, seed=0)
        assert abs(float(mask.float().mean()) - keep) < 0.03


def test_active_unit_occlusion_drops_only_active_units():
    """Zeros stay known; that is what makes this the harder difficulty scale.

    If this regressed to occluding zeros too, the surviving cue would keep more
    real signal than the reported `kept` fraction claims, and every delta in
    VIII.2 would be measured against an easier task than it says.
    """
    key = _sparse_key()
    active = key.abs() > 0
    mask = _occlude_active_units(key, 0.5, seed=0)
    assert bool(mask[~active].all()), "an inactive coordinate was occluded"
    kept_active = float(mask[active].float().mean())
    assert 0.3 < kept_active < 0.7


def test_active_unit_occlusion_is_monotone_in_keep():
    """A smaller `keep` must never leave more signal than a larger one."""
    key = _sparse_key(n=2000, active=400)
    surviving = [
        float((key * _occlude_active_units(key, k, seed=0)).abs().gt(0).sum())
        for k in KEEPS
    ]
    assert surviving == sorted(surviving, reverse=True), surviving


def test_occlusion_is_deterministic_for_a_seed():
    key = _sparse_key()
    for occlude in (_occlude_coordinates, _occlude_active_units):
        a = occlude(key, 0.3, seed=7)
        b = occlude(key, 0.3, seed=7)
        assert torch.equal(a, b)


def test_cosine_arm_scores_the_degraded_cue_not_the_clean_key():
    """The baseline VIII.2 rests on, reproduced exactly as `measure_arm` scores it.

    `measure_arm` computes `(partial @ patterns.T).argmax() == target`. Scoring
    the *clean* key instead would be trivially perfect at every degradation and
    would silently turn the comparison into completion-versus-nothing -- so this
    asserts the degraded cue can actually miss.
    """
    g = torch.Generator().manual_seed(0)
    patterns = torch.randn(64, 200, generator=g)
    patterns = patterns / patterns.norm(dim=1, keepdim=True)
    target = 5
    key = patterns[target]

    clean_hit = int((key @ patterns.T).argmax()) == target
    assert clean_hit, "the clean key must retrieve itself"

    # At severe occlusion the same arm must be capable of missing; if it never
    # misses, it is not measuring what VIII.2 claims it measures.
    misses = 0
    for seed in range(40):
        mask = _occlude_coordinates(key, 0.02, seed=seed)
        partial = key * mask
        if float(partial.norm()) < 1e-8:
            continue
        misses += int(int((partial @ patterns.T).argmax()) != target)
    assert misses > 0
