"""Regression guards for issue 03 — the rulebook harness's two honesty defects.

`experiments/rulebook_eval.py` reports two numbers that used to be
unfalsifiable:

  1. `stale@256` for the ingestion-gated arm was computed by ranking only the
     rules the gate decided to keep (`knn_arm(live, embedder)`). A rule the
     gate dropped could not structurally appear in that ranking, so a gate
     that removed every superseded rule scored a *guaranteed* 0.00 regardless
     of how retrieval would actually have ranked those rules. The fix ranks
     over the full rule set and expresses the gate's verdict as a demotion,
     not a deletion, so a rule the gate *misses* keeps its ordinary rank and
     is still exposed to genuine staleness risk.
  2. The token-budget loop in `evaluate()` used `continue` when a rule didn't
     fit, so it kept descending the ranking and backfilled with cheaper,
     lower-ranked rules instead of stopping. The fix `break`s at the first
     rule that doesn't fit, and `evaluate()` now reports the mean number of
     rules actually admitted.

These tests use a tiny deterministic fake embedder (exact, controllable
cosines via an orthonormal basis) rather than the real BGE runtime, so they
run fast and do not depend on the optional embedding runtime.
"""

from __future__ import annotations

import torch

from experiments import rulebook_eval
from experiments.rulebook import Rule, Situation


# --------------------------------------------------------------- fake embedder


class _FakeEmbedder:
    """Looks text up in a fixed table instead of encoding it. `encode` and
    `encode_query` are the same lookup -- these tests aren't exercising query/
    document asymmetry, just the ranking and budget logic downstream."""

    def __init__(self, table: dict[str, torch.Tensor]):
        self._table = table

    def encode(self, texts):
        return torch.stack([self._table[t] for t in texts])

    def encode_query(self, texts):
        return self.encode(texts)


def _basis(n: int, i: int) -> torch.Tensor:
    v = torch.zeros(n)
    v[i] = 1.0
    return v


def _make_pair_rules():
    """Two independent supersession pairs in a 4-D orthonormal basis, so the
    cosine between any two vectors is exact and controllable:

      old1 -> new1 at cosine 0.95 (above a 0.9 gate threshold: caught)
      old2 -> new2 at cosine 0.50 (below a 0.9 gate threshold: missed)

    old1/old2 are orthogonal to each other's pair, so the gate's nearest-
    match search cannot cross-wire the two chains.
    """
    e0, e1, e2, e3 = (_basis(4, i) for i in range(4))
    theta = torch.acos(torch.tensor(0.95))
    new1 = torch.cos(theta) * e0 + torch.sin(theta) * e1
    phi = torch.acos(torch.tensor(0.50))
    new2 = torch.cos(phi) * e2 + torch.sin(phi) * e3

    rules = [
        Rule("old1", "old1", "test", 0),
        Rule("old2", "old2", "test", 1),
        Rule("new1", "new1", "test", 50, supersedes="old1"),
        Rule("new2", "new2", "test", 51, supersedes="old2"),
    ]
    table = {"old1": e0, "old2": e2, "new1": new1, "new2": new2}
    return rules, _FakeEmbedder(table)


# ------------------------------------------------------- defect 1: gated_arm


def test_gated_arm_ranks_over_the_full_rule_set_not_the_shrunk_store():
    """An earlier version called `knn_arm(live, embedder)`, so a rule the gate
    dropped could not appear in the ranking at all -- `set(ordering)` would
    equal only the 3 kept ids, not all 4. The fix ranks over every rule."""
    rules, embedder = _make_pair_rules()
    rank = rulebook_eval.gated_arm(rules, embedder, threshold=0.9)
    ordering = rank("old2")

    assert set(ordering) == {"old1", "old2", "new1", "new2"}, (
        "staleness must be scored against the full rule set, not a store "
        "the gate already shrunk"
    )
    # old1 -> new1 is at cosine 0.95, above the 0.9 threshold: the gate
    # correctly replaces it, so old1 is demoted after every kept rule.
    assert ordering.index("old1") == len(ordering) - 1
    # old2 -> new2 is at cosine 0.50, below the 0.9 threshold: the gate
    # misses it, so old2 keeps its ordinary similarity rank. The query here
    # is "old2" itself, so it must still come out on top -- a missed
    # supersession is fully exposed to retrieval, not hidden by construction.
    assert ordering[0] == "old2"


def test_gate_that_drops_nothing_scores_exactly_like_plain_knn():
    """A gate with an unreachable threshold drops nothing. Its ranking must
    be identical to plain kNN over the same rules -- i.e. it must score
    exactly as badly as the ungated arm, not benefit from any structural
    advantage the harness gives the gated arm."""
    rules, embedder = _make_pair_rules()
    gated = rulebook_eval.gated_arm(rules, embedder, threshold=2.0)  # > any cosine
    plain = rulebook_eval.knn_arm(rules, embedder)
    assert gated("old2") == plain("old2")


def test_stale_at_256_is_nonzero_when_the_gate_misses_a_supersession(monkeypatch):
    """End-to-end through evaluate(): a gate that fails to catch a real
    supersession must show up as a nonzero stale@256, because the missed
    rule remains fully rankable. A gate that catches every supersession in
    the same corpus, under the same budget, must score 0."""
    rules, embedder = _make_pair_rules()
    tokens = {r.rule_id: 80 for r in rules}
    situation = Situation("old2", gold=("new2",), traps=())
    monkeypatch.setattr(rulebook_eval, "SITUATIONS", [situation])
    monkeypatch.setattr(rulebook_eval, "BUDGET", 170)  # room for exactly 2 rules
    monkeypatch.setattr(rulebook_eval, "stale_rule_ids", lambda: {"old1", "old2"})

    missed = rulebook_eval.evaluate(
        "misses old2", rulebook_eval.gated_arm(rules, embedder, threshold=0.9),
        tokens, verbose=False,
    )
    assert missed["stale"] > 0.0

    caught = rulebook_eval.evaluate(
        "catches everything", rulebook_eval.gated_arm(rules, embedder, threshold=0.4),
        tokens, verbose=False,
    )
    assert caught["stale"] == 0.0


# ------------------------------------------------------- defect 2: budget stop


def test_budget_stops_at_the_limit_instead_of_backfilling(monkeypatch):
    """The first ranked rule alone exceeds the budget. The loop must stop
    there and admit nothing, not skip it and pack in the cheaper rules
    ranked behind it."""
    tokens = {"expensive": 300, "cheap_a": 5, "cheap_b": 5}
    ranker = lambda question: ["expensive", "cheap_a", "cheap_b"]  # noqa: E731
    situation = Situation("q", gold=("cheap_a",), traps=())
    monkeypatch.setattr(rulebook_eval, "SITUATIONS", [situation])
    monkeypatch.setattr(rulebook_eval, "stale_rule_ids", lambda: set())

    result = rulebook_eval.evaluate("stop-at-budget", ranker, tokens, verbose=False)

    assert result["admitted"] == 0.0
    assert result["coverage"] == 0.0


def test_budget_admits_a_cheaper_rule_only_when_it_is_ranked_ahead(monkeypatch):
    """Sanity check on the same fixture: if the cheap rules are ranked first,
    stopping at the limit still admits them -- the fix changes *when* the
    loop stops, not whether affordable, correctly-ranked rules get in."""
    tokens = {"expensive": 300, "cheap_a": 5, "cheap_b": 5}
    ranker = lambda question: ["cheap_a", "cheap_b", "expensive"]  # noqa: E731
    situation = Situation("q", gold=("cheap_a",), traps=())
    monkeypatch.setattr(rulebook_eval, "SITUATIONS", [situation])
    monkeypatch.setattr(rulebook_eval, "stale_rule_ids", lambda: set())

    result = rulebook_eval.evaluate("cheap-first", ranker, tokens, verbose=False)

    assert result["admitted"] == 2.0
    assert result["coverage"] == 1.0
