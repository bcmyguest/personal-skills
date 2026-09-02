# 13 — Key/value superposition with a learned read-in

**What to build:** The compression claim gets its fair test.

Part VII measured superposed memories injected into a frozen decoder's
attention and found that compressing rules into a single slot recovers none of
the benefit — while confirming exactly that the same content supplied as
prefilled cache rather than prompt text costs zero context tokens. Three things
make the negative half of that weaker than it reads: the model's query
projections were never trained to read a memory slot, the compression was crude
mean-pooling rather than anything learned, and whitening — the fix that made
superposition work in embedding space — cannot be applied inside a frozen model
without breaking the scores that would use it.

One conceptual correction belongs in this ticket's write-up: **"N memories into
one vector" is not a Hopfield capacity claim.** Hopfield capacity concerns
storing N patterns as N separate rows and retrieving one; it says nothing about
summing them. The claim belongs to the vector-symbolic / holographic-reduced-
representation literature, which sets capacity for reliable unbinding far below
what Part V's framing implies and requires near-orthogonal codes — which is
what the whitening result rediscovered. The project should cite the literature
that actually governs the claim.

**Blocked by:** 09 — Separation × inverse-temperature sweep (if separation
turns out not to matter for retrieval, the geometry premise here is worth
revisiting before spending on a trained read-in).

**Status:** ready-for-agent

- [ ] A learned projection maps superposed memories into the geometry the
      model's queries expect, replacing frozen mean-pooling
- [ ] The compression curve from Part VII is re-run against it, using the same
      metric and the same matched-noise controls, so the numbers are comparable
- [ ] The zero-context-token result is re-confirmed as the separate, already
      exact claim it is, and kept distinct from the compression claim
- [ ] Part V.4's cost table is republished with a ratio the measurement
      supports, and the capacity claim is re-grounded in the correct literature
