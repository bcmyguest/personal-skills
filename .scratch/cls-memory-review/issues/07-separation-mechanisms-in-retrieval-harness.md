# 07 — Separation mechanisms wired into the retrieval harness

**What to build:** The retrieval ablation can express the configurations the
Hopfield capacity results actually describe. Right now it cannot: whitening
appears nowhere in the retrieval harness at all, so no whitened retrieval
number has ever been produced. Pattern separation is reachable but is not part
of any published sweep.

This is prefactoring, not a result. It makes tickets 08 onward possible, and on
its own it delivers one thing: the harness can emit a whitened row and a
separated row alongside the existing baseline, on both corpora.

**Blocked by:** 02 — Symmetric query encoding for the whitened embedder
(otherwise every whitened row it emits is confounded from the start).

**Status:** ready-for-agent

- [ ] The harness accepts whitening and key-representation as ordinary sweep
      axes, alongside the ones it already sweeps
- [ ] It runs on both corpora, per the project's standing rule that a change
      helping one and hurting the other is overfitting
- [ ] Each emitted row records the separation actually achieved, not just the
      configuration requested — the measured quantity, not the intent
- [ ] Baseline rows reproduce the currently published numbers unchanged
