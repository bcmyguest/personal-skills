# 08 — Turn the separation mechanisms on and re-measure

**What to build:** The system is measured with its separation mechanisms
enabled, which has never been done.

The reasoning matters more than the change. Hopfield capacity results are
conditional on patterns being well separated — the capacity term is multiplied
by the separation between a pattern and its neighbours. This project measured
that separation at roughly 0.29 and, instead of raising it, compensated by
raising the inverse temperature from 8 to 128. But the project separately
proved that at high inverse temperature the energy becomes a monotone function
of top-1 cosine similarity. **The compensation is the degeneracy.** Every
retrieval conclusion — including "the Hopfield layer ties cosine nearest
neighbours" — was measured in the one regime where that tie is guaranteed by
construction.

Two mechanisms in this codebase raise separation directly, and both are off by
default: whitening (measured to move anisotropy from +0.649 to −0.001) and
dentate-gyrus pattern separation. Turn them on and measure. This ticket is
deliberately the minimal version — flip the switches, report what happens, both
corpora — so that the sweep in 09 starts from a known point.

**Blocked by:** 07 — Separation mechanisms wired into the retrieval harness.

**Status:** ready-for-agent

- [ ] Recall is reported on both corpora for: baseline, whitening on, pattern
      separation on, and both on
- [ ] Achieved separation is reported per configuration, so the change in the
      quantity the theory cares about is visible next to the change in recall
- [ ] Whichever configuration wins, the defaults and their docstrings are
      updated to match the measurement, or the measurement is recorded as the
      reason for keeping them
- [ ] RESULTS gains a section stating plainly whether the earlier retrieval
      conclusions survive being measured outside the high-temperature regime
