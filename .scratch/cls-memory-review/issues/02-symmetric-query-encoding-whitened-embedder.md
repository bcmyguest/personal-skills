# 02 — Symmetric query encoding for the whitened embedder

**What to build:** Whitened and unwhitened retrieval arms are compared on equal
terms. The whitened embedder currently has no query-side encoding path, so when
it is measured through the dense ranker it silently loses the encoder's query
prefix while the baseline arm keeps it. Every whitened-versus-shipped number is
therefore partly measuring a missing prefix rather than whitening.

This is the same asymmetry that already forced one retraction in this project,
recurring in a new class.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] The whitened embedder exposes the same document/query encoding
      distinction as the embedder it wraps, and delegates the prefix correctly
- [ ] A test fails if the query path and document path are ever made identical
      again
- [ ] The whitening comparison in RESULTS Part VI.3 is re-measured and the
      published table updated (or confirmed unchanged, with the check recorded)
