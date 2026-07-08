Be sure to follow the following:

- Make a plan for larger changes then split it up into small, incremental changes and prompt the user to accept before moving forward
- If you don't know, say it. No sense in wasting time.
- Make sure to update documentation in CLAUDE.md and README.md
- Review your skills before starting tasks, you may need to trigger one
- Remember to add function/class definitions after constant definitions in the file

## Tooling

Use the `glab` CLI for all GitLab operations (MRs, pipelines, issues). Do NOT use MCP GitLab tools.

## Communication Style

When the user asks for a single command or concise answer, provide ONLY that command—no verbose multi-line snippets, warnings, or extra explanation unless asked.

## Documentation

Never fabricate or pad documentation; use only real data from the repo. If data is sparse, say so rather than filling template slots.

## Refactoring rules

Restate and confirm design constraints before editing. Do not change existing tests to match your assumptions—if a test seems wrong, ask first.

## Testing

After any code change, run the relevant tests and report pass/fail before committing.

## General Approach

- Prefer the smallest-scope change (e.g., an `.env.test` tweak) over modifying shared code/config; ask before making destructive or wide-ranging edits.
- Do not make confident claims you cannot verify. If uncertain about a technical detail (TLS behavior, encoding, library internals), say so and check rather than asserting.

## Debugging Workflow

- When investigating a bug, act quickly: reproduce once, form a hypothesis, and make a fix. Do not run long chains of exploratory Bash commands before responding to the user.
- When the user has already diagnosed the root cause, trust their diagnosis and pursue that path first rather than proposing alternative theories.
- Don't over-explore. If exploration is genuinely needed, spend some of that time updating docs/CLAUDE.md so future agents don't have to re-derive the same context.

## Python Conventions

- On "no record found", raise an exception rather than returning None.
- Prefer match/case over dispatch dicts when code needs to be mock-patchable in tests.

## Git & MR Workflow

- Keep commit messages short and concise.
- Remember that pushing a branch to GitLab may auto-create an MR—check before opening a new one.
