# Shared brief — cls-memory review tickets

Read this before touching anything. It is the same for every agent on this
review.

## Where you are

- Repo root: `/home/b/personal-skills`
- Project:   `/home/b/personal-skills/cls-memory`  ← **do all work from here**
- Branch:    `claude/cls-organizational-memory-poc-s9afk0`
- Tickets:   `/home/b/personal-skills/.scratch/cls-memory-review/issues/`

## Environment (already verified working — do not re-derive)

```bash
cd /home/b/personal-skills/cls-memory
uv run pytest -q -p no:warnings      # 151 passed, 0 skipped, ~27s
PYTHONPATH=. uv run python experiments/<script>.py
```

- The optional embedding runtime (`onnxruntime` + `tokenizers`) **is installed
  here**, and BGE weights are already on disk at
  `experiments/data/fast-bge-small-en-v1.5/`. Real-embedding tests run.
- **Both corpora are present**: `experiments/data/locomo10.json` and
  `experiments/data/qmsum_test.jsonl`. Load with `experiments.locomo.load()`
  and `experiments.qmsum.load(max_meetings=N)`.
- Network: PyPI, `raw.githubusercontent.com` (non-LFS), GitHub *release*
  assets and `storage.googleapis.com` are reachable. HuggingFace is **not**.

## Non-negotiable rules

1. **Never fabricate a number.** Every figure that lands in `RESULTS.md`,
   `README.md` or a docstring must come from a command you actually ran in
   this session. If a measurement is too expensive to run, say so in your
   report and leave the old number with an explicit note — do not estimate,
   interpolate, or "expect".
2. **Both corpora.** This project's standing rule: a change that helps LoCoMo
   and hurts QMSum is overfitting. Any recall claim needs both.
3. **The resolution limit is real.** At LoCoMo's n=494, differences below
   ~0.04 hit@1 are **not resolvable**. Use `experiments/metrics.py`'s exact
   paired McNemar test. Do not report a sub-threshold difference as a finding.
4. **hit@k, not recall@k.** 93 of 494 LoCoMo questions have multiple evidence
   turns. Say hit@k.
5. **Do not change an existing test to make your code pass.** If a test looks
   wrong, stop and report it. (See traps below — this project has been bitten
   three times by exactly that.)
6. **Verify by a route you did not use to build the thing.** "My fix + my
   test" is one unreviewed unit.
7. **Prefer probes that check values, not shapes and exit codes.** The three
   worst defects in this repo all reported success while returning wrong
   answers.
8. **Do not `git commit`.** The orchestrator commits. Leave your work in the
   working tree.
9. **RESULTS.md is shared with another agent running in parallel.** Only ever
   make *targeted* edits to the sections your ticket names. Never rewrite the
   whole file, never use Write on it, and re-read it immediately before each
   edit.

## Traps this project keeps falling into

- **Query/document asymmetry.** BGE v1.5 puts an instruction prefix on
  *queries only*. Wrappers that forget `encode_query` silently encode queries
  with the document path — this has already forced one retraction, and ticket
  02 is the same bug in a new class. Whenever you wrap an embedder, forward
  `encode_query`.
- **In-sample fits.** Fitting a whitener/IDF/SVD on the same vectors you then
  score is transductive and flatters the result. Fit scope was worth about
  half of one published loss here.
- **Randomised methods need mean ± sd over seeds.** A single seed picked the
  maximum of five draws once already.
- **Defaults go untested because every experiment overrides them.** `beta=8`
  collapsed retrieval completely and nobody noticed for weeks. If you add a
  knob, make at least one measurement run at its actual default.
- **Synthetic data flatters every mechanism.** recall@1 was 1.000 on the
  template corpus and 0.121 on real dialogue with identical code. A synthetic
  result is not evidence.
- **Degeneracy at high beta.** At `beta=128` the Hopfield energy is provably a
  monotone function of top-1 cosine (measured rank correlation 1.0000), so
  "the energy adds nothing over cosine" is arithmetic, not a finding. Several
  tickets exist precisely to re-measure outside that regime.

## What "done" means for your ticket

- Every checkbox in the ticket file is satisfied, or explicitly reported as
  not satisfied with the reason.
- `uv run pytest -q -p no:warnings` passes with **no new skips**.
- New behaviour has a test that would fail if the defect were reintroduced.
- Any published number you touched is either re-measured or explicitly marked
  as unverified.
- You report, in your final message: what you changed, the commands you ran,
  the numbers you got, and anything you could not do.

## Working discipline (added after wave 1 — read this)

Two agents were killed mid-task by a session limit in wave 1 and left the
working tree in half-finished states that cost real cleanup. Work so that
being killed at any moment is harmless:

- **Never leave the tree knowingly broken.** If you want to check that a new
  test really fails against the old buggy code, do NOT revert the fix in
  place. Copy the old version to `/tmp` and run against that, or use
  `git stash` only if you restore it in the *same* command. One agent died
  exactly between "reverted the fix" and "restored the fix".
- **Finish each edit completely before starting the next.** Do not leave a
  function defined in two places mid-move. Another agent died mid-refactor
  with a helper duplicated at module level and inside `main()`.
- **Land your work in small, complete steps**, each of which leaves the suite
  green, rather than one big sweep at the end.
- If you are running low on room, stop and report what is done and what is
  not — a truthful partial result is far more useful than an interrupted one.
