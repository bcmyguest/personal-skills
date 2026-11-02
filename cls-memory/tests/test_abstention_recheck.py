"""Regression guards for review ticket 12 -- the abstention grid.

`experiments/abstention_recheck.py` re-scores LoCoMo category-5 abstention
across a GRID of (separation arm x beta) instead of the single beta=128 cell
RESULTS.md IV.1 used, because that cell is exactly where ticket 09's sweep
shows `log_density` is *provably* a monotone function of `cos_top1` (rho ->
1.0000) -- so "the energy adds nothing" there is arithmetic, not a finding.

Three things are checked here, matching the ticket's checkboxes:

  1. `bonferroni_alpha` and `paired_auc_delta_ci` -- the new pure/generalised
     statistics functions -- are correct in isolation.
  2. `collect_grid` produces a `rho`-computable column set per beta cell (the
     "degeneracy is visible per row" checkbox) without disturbing the shape
     `abstention.py`'s existing AUC/CI/out-of-fold machinery expects.
  3. The out-of-fold split (`abstention.loco_auc`, reused unmodified) never
     trains a fold's model on the conversation it is about to score -- the
     property that makes "out-of-fold" mean something instead of leaking the
     conversation's own memory landscape into its own evaluation.

Runs on tiny synthetic data (a handful of turns, LSA-16) -- fast, no real
corpus needed, same pattern as `tests/test_recall_check.py`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import torch

from experiments import abstention
from experiments import locomo
from experiments.abstention_recheck import (
    ARMS,
    SIGNALS,
    bonferroni_alpha,
    collect_grid,
    paired_auc_delta_ci,
)
from experiments.metrics import spearman_rho

# --------------------------------------------------------------------- data

_SUBJECTS = ["Alice", "Bob", "Priya", "the vendor"]
_VERBS = ["booked", "reviewed", "cancelled", "rescheduled", "approved"]
_OBJECTS = [
    "the quarterly flight to Denver",
    "the marketing budget spreadsheet",
    "the vendor pricing contract",
    "the Thursday client call",
    "the conference hotel reservation",
    "the onboarding checklist",
]


def _make_conversation(sample_id: str = "synthetic-1", n_turns: int = 30,
                       offset_days: int = 0) -> locomo.Conversation:
    """A small lexically-varied conversation with BOTH answerable and
    category-5 adversarial questions, so a collected cell has both labels
    present (required for AUC to be defined at all).

    The adversarial question's evidence points at a real, resolvable turn --
    same shape as LoCoMo category 5: the near-miss is genuinely in memory,
    only the attribution is wrong, which is what makes it a hard negative
    rather than an easy out-of-distribution one.
    """
    base = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset_days)
    turns = []
    for i in range(n_turns):
        subject = _SUBJECTS[i % len(_SUBJECTS)]
        verb = _VERBS[(i // len(_SUBJECTS)) % len(_VERBS)]
        obj = _OBJECTS[i % len(_OBJECTS)]
        turns.append(
            locomo.Turn(
                dia_id=f"{sample_id}:d{i}",
                speaker=subject,
                text=f"{verb} {obj} on day {i}",
                session="s1",
                timestamp=base + timedelta(hours=i),
            )
        )
    questions = [
        locomo.Question(
            question="What did the vendor do about the pricing contract?",
            answer=turns[2].text,
            category=1,
            evidence=[turns[2].dia_id],
        ),
        locomo.Question(
            question="What did Alice realize after her own budget review?",
            answer="",  # deliberately empty -- category 5's defining shape
            category=5,
            evidence=[turns[1].dia_id],  # the near-miss turn IS in memory
        ),
    ]
    return locomo.Conversation(sample_id=sample_id, turns=turns, questions=questions)


# ------------------------------------------------------------- pure helpers


def test_bonferroni_alpha_divides_evenly():
    assert bonferroni_alpha(6) == pytest.approx(0.05 / 6)
    assert bonferroni_alpha(1, alpha=0.05) == pytest.approx(0.05)


def test_bonferroni_alpha_rejects_empty_family():
    """A grid with zero comparisons (empty --betas or --arms) is a caller
    bug; silently returning alpha=0.05 would hide it instead of failing."""
    with pytest.raises(ValueError):
        bonferroni_alpha(0)


def test_paired_auc_delta_ci_matches_original_at_alpha_05():
    """The generalised percentile-CI helper must reproduce
    `abstention.paired_auc_delta`'s hardcoded 95% CI exactly at alpha=0.05 --
    it is a strict generalisation, not a reimplementation that happens to
    agree most of the time."""
    torch.manual_seed(0)
    a = [float(x) for x in torch.rand(60)]
    b = [float(x) for x in torch.rand(60)]
    labels = [i % 3 != 0 for i in range(60)]  # deterministic mixed labels
    labels = [int(v) for v in labels]

    gen_a = torch.Generator().manual_seed(123)
    gen_b = torch.Generator().manual_seed(123)

    point1, lo1, hi1 = abstention.paired_auc_delta(a, b, labels, draws=500, generator=gen_a)
    point2, lo2, hi2 = paired_auc_delta_ci(a, b, labels, alpha=0.05, draws=500, generator=gen_b)

    assert point1 == pytest.approx(point2)
    assert lo1 == pytest.approx(lo2)
    assert hi1 == pytest.approx(hi2)


def test_paired_auc_delta_ci_narrows_as_alpha_shrinks():
    """A Bonferroni-corrected (smaller) alpha must widen, not narrow, the CI
    -- shrinking alpha means demanding more confidence, which can only make
    the interval wider or equal."""
    torch.manual_seed(0)
    a = [float(x) for x in torch.rand(80)]
    b = [float(x) for x in torch.rand(80)]
    labels = [int(i % 4 != 0) for i in range(80)]

    _, lo_wide, hi_wide = paired_auc_delta_ci(
        a, b, labels, alpha=0.05, draws=500, generator=torch.Generator().manual_seed(7)
    )
    _, lo_narrow, hi_narrow = paired_auc_delta_ci(
        a, b, labels, alpha=0.001, draws=500, generator=torch.Generator().manual_seed(7)
    )
    assert lo_narrow <= lo_wide
    assert hi_narrow >= hi_wide


# ------------------------------------------------------------- collect_grid


def test_collect_grid_reports_rho_computable_columns_per_beta():
    """Ticket 12's required checkbox: rho(signal, cos_top1) must be
    computable from every returned cell, for every beta, so degeneracy is
    visible per row rather than only at one hand-picked operating point."""
    convs = [_make_conversation("c0"), _make_conversation("c1", offset_days=30)]
    label, key_mode, whiten = ARMS[0]  # baseline: key=EMBEDDING, no whiten
    betas = (8.0, 128.0)

    cells = collect_grid(convs, dim=16, key_mode=key_mode, whiten=whiten, betas=betas)

    assert set(cells) == set(betas)
    for beta, cell in cells.items():
        cols = cell["columns"]
        n = len(cell["labels"])
        assert n > 0
        for signal in SIGNALS:
            assert len(cols[signal]) == n
        assert len(cols["length"]) == n
        # both classes present, or the AUC this feeds is undefined
        assert set(cell["labels"]) == {0, 1}
        for signal in SIGNALS:
            if signal == "cos_top1":
                continue
            rho = spearman_rho(
                torch.tensor(cols[signal], dtype=torch.double),
                torch.tensor(cols["cos_top1"], dtype=torch.double),
            )
            assert rho == rho or n < 2  # not NaN, unless degenerate (n<2)
            assert -1.0 - 1e-9 <= rho <= 1.0 + 1e-9


def test_collect_grid_high_beta_is_closer_to_degenerate_than_low_beta():
    """Structural sanity check on the grid's whole premise: rho(log_density,
    cos_top1) should move TOWARD 1.0 as beta rises (RESULTS.md IV.1's
    mechanism -- the mixture density is dominated by its nearest component at
    high beta). If this direction ever reversed on real data it would mean
    the grid's beta wiring is broken, not that the physics changed."""
    convs = [_make_conversation("c0", n_turns=40)]
    label, key_mode, whiten = ARMS[0]
    betas = (2.0, 512.0)

    cells = collect_grid(convs, dim=16, key_mode=key_mode, whiten=whiten, betas=betas)

    def rho_at(beta):
        cols = cells[beta]["columns"]
        return abs(spearman_rho(
            torch.tensor(cols["log_density"], dtype=torch.double),
            torch.tensor(cols["cos_top1"], dtype=torch.double),
        ))

    assert rho_at(512.0) >= rho_at(2.0) - 1e-6


def test_collect_grid_covers_the_three_negative_control_arms():
    """`ARMS` must be exactly the ticket's negative-control triple -- not
    ticket 09's fourth ("both") arm, which this recheck's design excludes."""
    labels = {label for label, _, _ in ARMS}
    assert labels == {"baseline", "whitened", "DG (neg control)"}


# ----------------------------------------------------------- out-of-fold


def test_loco_auc_never_trains_a_fold_on_its_own_held_out_conversation(monkeypatch):
    """The load-bearing property of `abstention.loco_auc`, reused unmodified
    by the grid: the model that scores conversation X's questions must never
    have been fit on conversation X's rows. Otherwise the embedder/store's own
    landscape leaks into its own evaluation and "out-of-fold" is a label, not
    a guarantee.

    Verified by capturing every `fit_logistic` call's training labels and
    checking they exactly equal "every label except this held-out
    conversation's", computed independently of `loco_auc`'s own masking.
    """
    conv_ids = [0, 0, 0, 1, 1, 1, 1, 2, 2]
    labels = [1, 0, 1, 1, 0, 1, 0, 0, 1]
    # A feature is required for the call shape; its value is irrelevant here.
    feature = [float(i) for i in range(len(conv_ids))]
    data = {"columns": {"cos_top1": feature}, "labels": labels, "conv": conv_ids}

    captured_y: list[torch.Tensor] = []

    def fake_fit_logistic(x: torch.Tensor, y: torch.Tensor, **kwargs):
        captured_y.append(y.clone())
        return torch.zeros(x.shape[1] + 1, dtype=x.dtype)

    monkeypatch.setattr(abstention, "fit_logistic", fake_fit_logistic)

    abstention.loco_auc(data, ["cos_top1"])

    held_ids = sorted(set(conv_ids))
    assert len(captured_y) == len(held_ids)
    for held, y_train in zip(held_ids, captured_y):
        expected = [lab for lab, cv in zip(labels, conv_ids) if cv != held]
        assert y_train.tolist() == [float(v) for v in expected], (
            f"fold held={held} trained on a label set that does not exactly "
            "exclude the held-out conversation"
        )
