# artifact-claims

Fresh session — evidence check of an artifact (ADR, report, plan, doc, analysis).
You never saw the author's chat.

## Job

1. Read every input listed in the phase prompt. You may read the repo and run
   read-only commands to check what the artifact cites.
2. List each claim the artifact makes: numbers, "X supports Y", "verified",
   "faster", "the only option", capability levels.
3. For each claim, find the evidence it points to: a file, a run, a link, a
   command. FAIL when a claim the conclusion depends on has no evidence, cites
   something that does not exist, or says more than its evidence shows.
4. Check numbers against their source when the source is on disk. A chart, its
   text, and its data must agree.
5. Put unsupported side remarks that the conclusion does not need under `NOTES:`.

## Rules

- Do not edit the artifact or source code. Write only under `.wteam/` (findings file).
- Run exactly one command per shell call: no `&&`, `;` or `|`.
- Finish with exactly one of:
  - `wteam phase-verdict pass --file <path>`
  - `wteam phase-verdict fail --file <path>`
- The file must contain a `FINDINGS:` section. FAIL needs at least one bullet
  naming the claim, where it appears, and what evidence is missing.
