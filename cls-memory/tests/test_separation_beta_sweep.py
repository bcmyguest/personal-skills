"""Guards for ticket 09 -- the separation x inverse-temperature grid.

`experiments/separation_beta_sweep.py` takes one shortcut that the rest of the
harness does not: it builds each separation arm ONCE and sweeps every beta
against that single store, on the argument that beta is a read-time parameter
and the write path cannot see it. That shortcut is what makes the full
cross-product affordable on QMSum (2 builds instead of 20), and if it is wrong
every number in the grid is wrong in a way no amount of staring would reveal.

So the load-bearing test here is a fidelity test: for the same arm and the same
beta, the swept cell must produce the *same per-question hit/miss vector* as
`recall_check.evaluate`, which rebuilds from scratch per cell. Not the same
aggregate -- the same 0/1 per question. Two runs can agree on hit@1 to three
decimals while disagreeing about which questions they got right.

Also pins `spearman_rho`, which is new and which the grid's central claim
("rho below 1.0 means the layer is not cosine kNN") rests on entirely.
"""

from __future__ import annotations

import pytest
import torch

from cls_memory import HippocampalKey
from experiments.metrics import spearman_rho
from experiments.recall_check import KS, SEED, evaluate
from experiments.separation_beta_sweep import sweep_arm
from test_recall_check import _make_conversation

# Small enough to run in seconds; large enough that LSA is not degenerate and
# the questions have somewhere to go wrong. Matches test_recall_check's shape.
DIM = 64
BETA = 8.0


def test_sweep_reproduces_evaluate_per_question() -> None:
    """The build-once/sweep-beta shortcut must be exactly the long way round."""
    convs = [_make_conversation()]

    torch.manual_seed(SEED)
    reference = evaluate(convs, dim=DIM, key_mode=HippocampalKey.EMBEDDING, beta=BETA)

    torch.manual_seed(SEED)
    swept = sweep_arm(
        convs,
        dim=DIM,
        key_mode=HippocampalKey.EMBEDDING,
        whiten=False,
        betas=(BETA,),
    )
    cell = swept["betas"][BETA]

    assert swept["asked"] == reference["asked"] > 0
    for k in KS:
        assert cell["per_question"][k] == reference["per_question"][k], (
            f"hit@{k} disagrees per question; the swept store is not the "
            "store evaluate() builds"
        )
        assert cell[k] == reference[k]
    # The separation actually achieved is measured over the same written keys.
    assert swept["aniso_emb"] == reference["aniso_emb"]
    assert swept["aniso_key"] == reference["aniso_key"]


def test_sweep_reproduces_evaluate_whitened() -> None:
    """Same fidelity check on the whitening arm, where the embedder is wrapped.

    Whitening is fitted once on the whole corpus outside the per-conversation
    loop; a sweep that refitted it per arm-and-beta, or forgot to fit it at
    all, would show up here as a different per-question vector.
    """
    convs = [_make_conversation()]

    torch.manual_seed(SEED)
    reference = evaluate(
        convs, dim=DIM, key_mode=HippocampalKey.EMBEDDING, beta=BETA, whiten=True
    )

    torch.manual_seed(SEED)
    swept = sweep_arm(
        convs, dim=DIM, key_mode=HippocampalKey.EMBEDDING, whiten=True, betas=(BETA,)
    )

    assert swept["betas"][BETA]["per_question"][1] == reference["per_question"][1]
    assert swept["aniso_emb"] == reference["aniso_emb"]


def test_cosine_reference_does_not_depend_on_beta() -> None:
    """The comparator is plain cosine kNN: no settling, no temperature.

    It is measured inside the beta loop's parent scope, so a refactor that
    accidentally let a beta leak into it would silently turn the baseline into
    a second Hopfield arm -- and every 'competitive with cosine' verdict in the
    grid would be comparing the layer against itself.
    """
    convs = [_make_conversation()]

    torch.manual_seed(SEED)
    low = sweep_arm(
        convs, dim=DIM, key_mode=HippocampalKey.EMBEDDING, whiten=False, betas=(2.0,)
    )
    torch.manual_seed(SEED)
    high = sweep_arm(
        convs, dim=DIM, key_mode=HippocampalKey.EMBEDDING, whiten=False, betas=(512.0,)
    )

    for k in KS:
        assert low["cosine"]["per_question"][k] == high["cosine"]["per_question"][k]


def test_grid_reports_the_rank_correlation_per_cell() -> None:
    """Every cell carries rho_rank, top-1 agreement and a McNemar against cosine.

    These are the ticket's deliverables, not decoration: without the paired
    test a cell inside the noise floor reads as a finding.
    """
    convs = [_make_conversation()]
    torch.manual_seed(SEED)
    swept = sweep_arm(
        convs,
        dim=DIM,
        key_mode=HippocampalKey.EMBEDDING,
        whiten=False,
        betas=(2.0, 128.0),
    )

    for beta in (2.0, 128.0):
        cell = swept["betas"][beta]
        assert -1.0 <= cell["rho_rank"] <= 1.0
        assert cell["rho_rank_n"] > 0
        assert 0.0 <= cell["top1_agree"] <= 1.0
        assert 0.0 <= cell["mixture"] <= 1.0
        for k in KS:
            assert 0.0 <= cell[f"mcnemar@{k}"]["p"] <= 1.0


def test_spearman_matches_hand_computed_values() -> None:
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # d = [-1, 1, -1, 1, 0] -> 1 - 6*4/(5*24) = 0.8
    assert spearman_rho(a, torch.tensor([2.0, 1.0, 4.0, 3.0, 5.0])) == pytest.approx(0.8)
    assert spearman_rho(a, a * 3.0 + 1.0) == pytest.approx(1.0)
    assert spearman_rho(a, -a) == pytest.approx(-1.0)


def test_spearman_is_undefined_not_zero_for_a_constant_input() -> None:
    """A constant vector has no ranking. Returning 0.0 would read as a
    measured null and would quietly average into a cell's rho_rank."""
    rho = spearman_rho(torch.ones(5), torch.arange(5.0))
    assert rho != rho  # NaN


def test_spearman_tie_corrects() -> None:
    """Tied groups take the average rank; without that the correlation is
    inflated by whatever order argsort happened to pick."""
    tied = torch.tensor([1.0, 1.0, 2.0, 2.0])
    assert spearman_rho(tied, torch.tensor([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.8944271909999159)
    # Tie-broken in the opposite direction: same answer, because ranks are averaged.
    assert spearman_rho(tied, torch.tensor([2.0, 1.0, 4.0, 3.0])) == pytest.approx(0.8944271909999159)
