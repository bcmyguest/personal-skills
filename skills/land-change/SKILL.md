---
name: land-change
description: Get exactly the intended change onto the right branch, verified, with the evidence a reviewer needs. Use when asked to commit, push, open a PR or MR, merge once checks pass, or get work into main. Not for reviewing the diff itself (audit-agent-diff) or porting work between branches (port-changes).
---

# Land change

Commit only the intended change, push it, and carry it as far as the request authorises.

## Gather first

- `git status`, the branch, and its upstream. Whether main moved since the branch was cut.
- The request: commit only, push, PR, or merge after checks.
- Branch and CI policy from repo instructions.

## Method

1. **Stage selectively.** Leave out user-local files (`.env.*`), debug output, screenshots, generated artefacts, and work that belongs to another feature or another agent. List what was left out.
2. **Branch from fresh main** when asked; rebase before starting, not after conflicts pile up.
3. **Run the relevant tests** and report pass or fail before committing.
4. **Commit message:** short title that says what changed. Fold fixups into the commit they fix.
5. **PR evidence:** what changed, how it was verified, and a screenshot for UI changes, attached to the PR, not committed.
6. **Wait for CI** before merging. A red check is a stop, not a retry loop.
7. **Sync dependent copies** the repo documents (mirrors, vendored copies, a second remote).
8. **Finish.** An uncommitted or unpushed result is not done; report the commit hash, branch, and PR link.

## Boundaries

Push, PR, and merge are external actions: perform them only as far as the request says. Never force-push a shared branch or rewrite published history without an explicit request.

## Next

None: this ends the chain.
