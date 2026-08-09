# 03 — Rulebook evaluation reports honest coverage and staleness

**What to build:** The rulebook harness reports numbers that could come out
otherwise. Two defects currently prevent that:

1. **Staleness is circular.** The ingestion gate removes superseded rules from
   the store, and the evaluation then counts how many superseded rules were
   retrieved *from that same store*. The answer is structurally zero. The
   headline `stale@256 = 0.00` — the single number Part IV cites as the design's
   one real win — is arithmetic, not evidence.
2. **The token budget backfills.** When a rule does not fit the budget the loop
   skips it and keeps descending the ranking, quietly packing cheaper
   low-ranked rules. The documented behaviour ("the budget admits about seven
   rules") is not what is measured.

After this ticket the gating result is either confirmed against a store that
still contains the superseded rules, or it is retracted.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Staleness is scored against the full rule set, so a gate that removes
      nothing scores badly and a gate that works scores well
- [ ] The budget stops at the limit rather than backfilling, and the number of
      rules actually admitted is reported alongside coverage
- [ ] A test would fail if either defect were reintroduced
- [ ] RESULTS Part IV.2's table is republished with the corrected numbers, and
      the write-time gating conclusion is restated, revised, or withdrawn to
      match what the corrected numbers support
