# 11 — Gist recall in the metastable regime

**What to build:** The project's third headline claim — schema absorption, the
"complementary learning systems" idea the whole design is named for — gets
measured for the first time.

The library already exposes a deliberately low-temperature recall path that
settles into a metastable mixture rather than a single episode, and the theory
describes exactly this regime: at intermediate inverse temperature, related
patterns merge into a metastable state that averages over a cluster. That is
what a *schema* would look like mechanically, and it is the one thing in this
architecture a vector database has no analogue for.

It has never been evaluated. The low-temperature regime was dismissed once, on
an adversarial question set built from near-verbatim paraphrases, where density
provably cannot help — a task that could not have shown a gist effect even if
one existed.

**Blocked by:** 09 — Separation × inverse-temperature sweep (which identifies
the metastable band worth probing).

**Status:** ready-for-agent

- [ ] Gist recall is evaluated on a task where a correct answer requires
      generalising across several related memories, not identifying one
- [ ] Its output is compared against both single-episode recall and a
      centroid-of-top-k baseline, so any advantage is attributed to the
      metastable dynamics rather than to averaging as such
- [ ] The claim "schema absorption: not yet tested" in the status table is
      replaced with a measured verdict
