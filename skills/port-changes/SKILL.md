---
name: port-changes
description: Bring selected work from another branch, PR, archive, or older architecture into the current line without dragging unwanted changes along. Use for "take X from branch Y", "incorporate PR N", "what is in this tarball that is not in the branch", "recover the old branch into the new layout", or splitting a branch into reviewable PRs. Not for landing the result (land-change).
---

# Port changes

Inventory the delta, apply only the selected part, and list what was left behind.

## Gather first

- Source and target (branch, PR, archive, commit range) and the merge base.
- The user's selection rule: named files, a feature, "everything except X".

## Method

1. **Summarise the delta before applying:** what differs, grouped by feature, and whether each group touches code the target also changed.
2. **Apply only the selection.** Prefer `git cherry-pick`, `git checkout <src> -- <path>`, or a patch over retyping.
3. **Resolve conflicts by intent.** When a collaborator's version and the target conflict, keep the collaborator's intent unless told otherwise, and note each such choice.
4. **Translate across paradigms** when the source predates a restructure: map old modules to their new homes instead of reviving old files.
5. **Split along natural boundaries** (frontend and infra, schema and code) when the goal is reviewable PRs.
6. Run the target's tests on the result.

## Deliverable

Delta inventory, the applied subset, an explicit "left out" list with reasons, and test results.

## Boundaries

Edit the working tree. Pushing and opening PRs go through land-change.

## Next

- `audit-agent-diff` (transition): the applied subset and the left-out list.
