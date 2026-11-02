# 01 — Optional-dependency skips fail loudly

**What to build:** The test suite can no longer report success while silently
skipping the tests that hold the project's headline claims in place. Today,
without the optional embedding runtime installed, 15 tests skip and pytest
prints green — and those 15 are exactly the real-embedding tests that Parts V
and VI rest on. A reviewer sees "all tests pass" and has verified none of the
claims.

Running the suite in an incomplete environment must be visibly different from
running it in a complete one, at a glance, without reading the skip list.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] With the embedding runtime absent, the suite's exit status or summary
      makes it obvious the claim-bearing tests did not run
- [ ] With it present, the suite reports the full count (151 at time of writing)
      and no claim-bearing test is skipped
- [ ] The optional dependency needed to run the full suite is documented where
      someone setting up will actually see it
- [ ] CI (or the documented run command) exercises the complete path
