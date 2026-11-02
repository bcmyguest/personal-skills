# 05 — Lexical baseline and projection share one IDF convention

**What to build:** The lexical ceiling and the random projection are compared
under one inverse-document-frequency convention instead of two. The hook to
pass a shared corpus exists but nothing ever calls it, so the lexical baseline
is computed per-conversation while the projection uses corpus-wide statistics.

This matters because that comparison is what one of the project's published
retractions rests on, and the two arms differ by hundredths — well inside the
size of the confound.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Both arms are computed under the same IDF corpus, and which corpus is
      recorded in the output
- [ ] The comparison behind the retraction is re-run and the retraction is
      confirmed, revised, or itself withdrawn
- [ ] A run with mismatched IDF conventions is either impossible or loudly
      flagged
