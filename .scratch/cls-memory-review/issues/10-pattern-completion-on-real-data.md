# 10 — Pattern completion from degraded cues, on real data

**What to build:** The one retrieval behaviour Hopfield theory says a
nearest-neighbour lookup cannot match is measured on real text.

Iterative completion — settling from a partial or corrupted cue over several
updates — is where an attractor network is supposed to differ from a one-shot
similarity search. This project reports a strong completion result (recovery
from 26 of 205 active units) but measures it **only on the synthetic corpus**.
Part II already established that the synthetic retrieval numbers were mostly an
artifact of the generator, and retracted them on those grounds. The completion
number is the same class of number and has never faced the same test.

After this ticket, either completion holds up on real data — which would be the
first measured advantage the Hopfield layer has over cosine retrieval — or the
synthetic completion result joins the other retractions.

**Blocked by:** 08 — Turn the separation mechanisms on and re-measure (so
completion is measured at a separation the theory can work at).

**Status:** ready-for-agent

- [ ] Completion from degraded cues is measured on both real corpora across a
      range of cue degradation
- [ ] A single-shot nearest-neighbour arm is measured on identical cues, so the
      comparison is against the thing completion is meant to beat
- [ ] Iteration counts and convergence are reported, distinguishing "settled
      onto the right memory" from "settled onto one global mixture"
- [ ] RESULTS section 4 is republished or retracted against the real-data
      numbers
