# 04 — Whitener fitted on held-out data

**What to build:** The superposition capacity result stops being transductive.
The whitener is currently fitted on the same vectors it is then scored on, so
"whitening recovers the full theoretical capacity" is measured in-sample. Part
VI.3 already established that fit scope is a real artifact for ranking — it
found roughly half of one published loss was a refit artifact — but left the
capacity claim resting on an in-sample fit.

After this ticket the capacity numbers describe what a deployment would get,
where the whitener is fitted once on a corpus and then applied to vectors it
has never seen.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Capacity is measured with the whitener fitted on a disjoint corpus from
      the one superposed and decoded
- [ ] The in-sample and held-out numbers are both reported, so the size of the
      artifact is visible rather than assumed negligible
- [ ] RESULTS Part V.2's capacity table is republished against the held-out fit
- [ ] The fit-size warning threshold is checked against what the held-out
      measurement shows is actually needed
