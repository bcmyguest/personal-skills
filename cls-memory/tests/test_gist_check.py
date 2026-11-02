"""Fast tests for ticket 11's gist-recall harness (`experiments/gist_check.py`).

Tiny synthetic conversation, no real corpus needed -- same shape as
`test_recall_check._make_conversation`, but with one multi-evidence question
added: three scattered turns about the same topic ("the vendor pricing
contract"), which is the shape the actual task (LoCoMo/QMSum evidence sets
with >= 2 members) exercises on real data. What this file checks:

  1. the centroid-of-top-k baseline is genuinely dynamics-free -- it matches
     a hand-computed normalised mean and does not move when the network's
     settling config changes;
  2. the three arms are scored on exactly the same set of questions, not
     silently different subsets;
  3. the mixture diagnostics (top1 / effective_n) move in the direction the
     theory predicts between a very low and a very high beta.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import torch

from cls_memory import HopfieldConfig, ModernHopfieldNetwork
from experiments import locomo
from experiments.gist_check import _centroid_of_top_k, run

_TEXTS = [
    "Alice booked the marketing budget review for Monday.",
    "Bob reviewed the vendor pricing contract on Tuesday.",
    "Priya approved the vendor pricing renewal on Wednesday.",
    "Alice cancelled the Thursday client call.",
    "The vendor cancelled the onboarding session on Friday.",
    "Bob rescheduled the quarterly flight to Denver.",
    "Priya booked the conference hotel reservation.",
    "The vendor reviewed the pricing contract terms again on Saturday.",
    "Alice approved the marketing budget spreadsheet.",
    "Bob booked the Thursday client call.",
    "Priya reviewed the onboarding checklist.",
    "The vendor rescheduled the pricing contract signing.",
]
_SPEAKERS = ["Alice", "Bob", "Priya", "the vendor"]

# Turns 1, 7, 11 are the scattered "vendor pricing contract" family -- the
# multi-evidence question below needs all three, not just the nearest one.
_FAMILY = ("d1", "d7", "d11")


def _make_family_conversation() -> locomo.Conversation:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    turns = [
        locomo.Turn(
            dia_id=f"d{i}",
            speaker=_SPEAKERS[i % len(_SPEAKERS)],
            text=text,
            session="s1",
            timestamp=base + timedelta(hours=i),
        )
        for i, text in enumerate(_TEXTS)
    ]
    questions = [
        # Single-evidence question: an ordinary lookup, filtered OUT by
        # min_evidence=2 -- included to prove the filter actually filters.
        locomo.Question(
            question="What did Alice do on Monday?",
            answer=turns[0].text,
            category=1,
            evidence=["d0"],
        ),
        # Multi-evidence question: the task this harness is built to measure.
        locomo.Question(
            question="What happened with the vendor pricing contract, in total?",
            answer="reviewed, approved, reviewed again, rescheduled",
            category=1,
            evidence=list(_FAMILY),
        ),
    ]
    return locomo.Conversation(sample_id="family-1", turns=turns, questions=questions)


# ------------------------------------------------------------ arm 3: centroid


def test_centroid_is_dynamics_free():
    """`_centroid_of_top_k` is a plain mean, not one step of settling.

    It must reproduce a hand-computed normalised average of the nearest k
    patterns, and it must not move when `mhn.config` (beta, max_iter, ...)
    changes -- it never reads `config` at all.
    """
    torch.manual_seed(0)
    mhn = ModernHopfieldNetwork(dim=8, config=HopfieldConfig(beta=32.0, max_iter=32))
    patterns = torch.randn(6, 8)
    mhn.write(patterns)
    cue = torch.randn(8)

    k = 3
    got = _centroid_of_top_k(mhn, cue, k)

    sims = cue @ mhn.patterns.T
    idx = torch.topk(sims, k).indices
    expected = mhn.patterns[idx].mean(dim=0)
    expected = expected / expected.norm().clamp_min(1e-12)
    assert torch.allclose(got, expected, atol=1e-6)

    # Changing the settling config must not move the centroid: it is not
    # reachable through any config-dependent code path.
    mhn.config.beta = 0.001
    mhn.config.max_iter = 1
    still = _centroid_of_top_k(mhn, cue, k)
    assert torch.allclose(still, got, atol=1e-6)

    # And it must differ from what one settling step (`step`) would produce,
    # i.e. this is not secretly calling the attractor dynamics under another
    # name.
    settled_once = mhn.step(cue, beta=32.0)
    settled_once = settled_once / settled_once.norm().clamp_min(1e-12)
    assert not torch.allclose(got, settled_once, atol=1e-3)


# --------------------------------------------------------- identical queries


def test_three_arms_see_identical_queries():
    """recall, centroid and every gist row score exactly the same question set."""
    torch.manual_seed(0)
    conv = _make_family_conversation()
    result = run(
        [conv], dim=16, betas=(4.0, 64.0), factor=0.15, top_k=5, min_evidence=2,
    )

    # The single-evidence question must have been filtered out; only the
    # multi-evidence one remains.
    assert result["asked"] == 1
    assert result["recall"]["n"] == 1
    assert result["centroid_task"]["n"] == 1
    for label in result["labels"]:
        assert result["gist"][label]["n"] == 1
        assert result["centroid_diag"][label]["diag_n"] == 1


def test_min_evidence_filter_actually_filters():
    """With min_evidence=1 the single-evidence question is included too."""
    torch.manual_seed(0)
    conv = _make_family_conversation()
    result = run(
        [conv], dim=16, betas=(4.0,), factor=0.15, top_k=5, min_evidence=1,
    )
    assert result["asked"] == 2


# --------------------------------------------------------- beta sensitivity


def test_mixture_diagnostics_move_with_beta():
    """Very low beta should look more mixture-like than very high beta.

    This is the direction `hippocampus.step`'s docstring predicts even
    though it also predicts the mixture is not a *genuine* multi-memory
    fixed point: low beta pulls toward the broad, low-confidence global
    centroid (small top1, large effective_n); high beta snaps to one
    attractor (top1 -> 1, effective_n -> 1). The test only asserts the
    monotone direction, not that low beta is "correct" -- a negative result
    on the task metric is an acceptable outcome per the ticket.
    """
    torch.manual_seed(0)
    conv = _make_family_conversation()
    low_beta, high_beta = 0.5, 1.0e6
    result = run(
        [conv], dim=16, betas=(low_beta, high_beta), factor=0.15, top_k=5,
        min_evidence=2,
    )

    low = result["gist"][str(low_beta)]
    high = result["gist"][str(high_beta)]
    assert low["top1"] < high["top1"]
    assert low["effective_n"] > high["effective_n"]
    # At beta this extreme the settle is essentially a hard argmax: top1
    # should be right at the ceiling.
    assert high["top1"] > 0.99
