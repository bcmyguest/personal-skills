---
name: audit-agent-diff
description: Check an agent-written diff for the recurring ways agent changes overreach, before it is committed or opened as a PR. Use after an agent (including the current one) produced changes, or when asked to review what an agent did. Not general bug finding (Bugbot or a code-review skill) and not a security audit.
---

# Audit agent diff

Compare the diff against the request it serves and list overreach, each finding with its minimal correction.

## Gather first

- The full diff (`git diff` plus untracked files) and the request in the user's words.
- Repo conventions: instruction files, lint config, existing patterns for the same kind of change.

## Check for

1. **Invented machinery:** a new script, wrapper, or module where config, an existing command, or a package script suffices.
2. **Hardcoded values:** hosts, IPs, ports, image names, prices, URLs, model names that belong in env or config.
3. **Removed behaviour** the request did not ask to remove (a field, a badge, a config option, an error path).
4. **Tests:** an existing test deleted or weakened to pass is a blocker. A test the agent added in this change may be removed when it tests the wrong thing; say so.
5. **Scope creep:** edits to unrelated files, drive-by refactors, reformatting.
6. **Explicit constraints broken:** files the user said not to touch, dependencies they ruled out.
7. **Leaked values:** real keys, personal hosts, or local paths in examples, fixtures, or `.env.example`.
8. **Comment bloat:** comments that narrate the code or explain the change to a reviewer.

## Deliverable

Findings ordered by severity, each with file, line, and the minimal correction; or "clean" with the list of checks that ran. Say which checks were skipped (for example, no tests were run).

## Boundaries

Report. Apply corrections only when asked or when the audit runs inside a fix loop that owns the diff.

## Next

- `land-change` (transition): the clean diff.
- `bugbot` (alternative): general bug finding on the same diff.
