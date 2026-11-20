"""Regression guard for issue 07 -- separation mechanisms in the retrieval
harness.

`experiments/recall_check.py` sweeps default changes over two corpora to
check a change helps on both. Before this fix, whitening was reachable
nowhere in that harness -- no whitened retrieval number had ever been
produced -- and pattern separation, while reachable, was baked into one fixed
cumulative history rather than an independent sweep axis. Neither row
reported the separation it actually achieved; only hit@k.

These tests run on a tiny synthetic conversation (fast, no real corpus
needed) and check:

  1. `evaluate()` accepts `whiten` as an ordinary keyword alongside the
     existing `dim` / `key_mode` / `beta` axes, and runs both corpora's
     loader-shaped input (a `locomo.Conversation`; QMSum reuses the same
     dataclasses -- see `experiments/qmsum.py`).
  2. Enabling it measurably changes the isotropy of the embeddings actually
     written to the store -- not a config flag that does nothing. This is
     the "measured, not requested" checkbox: a row that requested whitening
     but silently got the pass-through (unfitted) embedder would show up
     here as `aniso_emb` close to the unwhitened baseline, not near zero.
  3. `key_mode` moves `aniso_key` (the separation actually achieved by the
     hippocampal key), independently of whether `whiten` touched `aniso_emb`
     -- the two axes are orthogonal, as the ticket requires.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cls_memory import HippocampalKey
from experiments import locomo
from experiments.recall_check import evaluate

# Small, lexically varied vocabulary so a low-dimensional LSA fit is not
# degenerate and embeddings for different turns are not near-identical.
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


def _make_conversation(n_turns: int = 36) -> locomo.Conversation:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    turns = []
    for i in range(n_turns):
        subject = _SUBJECTS[i % len(_SUBJECTS)]
        verb = _VERBS[(i // len(_SUBJECTS)) % len(_VERBS)]
        obj = _OBJECTS[i % len(_OBJECTS)]
        turns.append(
            locomo.Turn(
                dia_id=f"d{i}",
                speaker=subject,
                text=f"{verb} {obj} on day {i}",
                session="s1",
                timestamp=base + timedelta(hours=i),
            )
        )
    # One question anchored to an unambiguous, low-frequency turn.
    questions = [
        locomo.Question(
            question="What did the vendor do about the pricing contract?",
            answer=turns[2].text,
            category=1,
            evidence=[turns[2].dia_id],
        )
    ]
    return locomo.Conversation(sample_id="synthetic-1", turns=turns, questions=questions)


def _run(*, whiten: bool, key_mode: HippocampalKey = HippocampalKey.EMBEDDING) -> dict:
    conv = _make_conversation()
    return evaluate(
        [conv], dim=32, key_mode=key_mode, beta=32.0, whiten=whiten
    )


# ------------------------------------------------------------------- the axes


def test_evaluate_accepts_whiten_keyword():
    """`whiten` is an ordinary keyword, not a special-cased branch that only
    some callers can reach -- the same call shape as `dim`/`key_mode`/`beta`."""
    result = evaluate(
        [_make_conversation()],
        dim=32,
        key_mode=HippocampalKey.EMBEDDING,
        beta=32.0,
        whiten=True,
    )
    assert result["asked"] == 1
    assert set(result) >= {1, 5, 10, "asked", "aniso_emb", "aniso_key"}


def test_whitening_measurably_reduces_embedding_anisotropy():
    """The reported number reflects the vectors actually stored, not the
    request. If whitening silently no-op'd (e.g. the whitener was never
    fitted -- the exact trap `WhitenedEmbedder` warns about), `aniso_emb`
    would sit at the unwhitened baseline instead of dropping toward 0."""
    unwhitened = _run(whiten=False)
    whitened = _run(whiten=True)
    assert whitened["aniso_emb"] < unwhitened["aniso_emb"]
    # Whitening's ZCA transform is normalised so the result is isotropic in
    # the fitted subspace -- not just "lower", but close to 0.
    assert abs(whitened["aniso_emb"]) < 0.1


def test_key_mode_changes_key_anisotropy_independent_of_whitening():
    """Pattern separation (SEPARATED) acts on the hippocampal key, not the
    stored embedding -- so it should move `aniso_key` while leaving
    `aniso_emb` governed only by `whiten`, at matched `whiten`."""
    embedding_key = _run(whiten=False, key_mode=HippocampalKey.EMBEDDING)
    separated_key = _run(whiten=False, key_mode=HippocampalKey.SEPARATED)

    # aniso_emb depends only on whitening, not key_mode: both rows here are
    # unwhitened LSA-32 over the identical corpus, so the stored embeddings
    # (and hence their anisotropy) are the same regardless of key_mode.
    assert embedding_key["aniso_emb"] == separated_key["aniso_emb"]
    # EMBEDDING keys *are* the stored embedding, so aniso_key must equal
    # aniso_emb in that row; DG separation is a different projection and is
    # not required to land on the same value.
    assert embedding_key["aniso_key"] == embedding_key["aniso_emb"]
    assert separated_key["aniso_key"] != embedding_key["aniso_key"]
