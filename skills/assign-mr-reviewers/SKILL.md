---
name: assign-mr-reviewers
description: Assign reviewers to the user's active GitLab merge requests based on CODEOWNERS. Use this skill whenever the user wants to assign reviewers, add code owners, or get approvals on their MRs/merge requests. Also trigger when the user mentions "who should review", "need reviewers", "assign owners", or anything about getting their MRs reviewed.
---

# Assign MR Reviewers

Assign code owner reviewers to the user's open GitLab merge requests using `glab` and the bundled script.

The script is at: `~/.claude/skills/assign-mr-reviewers/scripts/assign_mr_reviewers.py`

## Important: How to run the script

The script must be run as a **standalone Bash command** with no piping, chaining, or redirection. This is required for auto-approve permissions to work.

- Run: `python3 ~/.claude/skills/assign-mr-reviewers/scripts/assign_mr_reviewers.py <args>`
- Do NOT: `python3 ~/.claude/skills/assign-mr-reviewers/scripts/assign_mr_reviewers.py <args> | jq .`

The script outputs JSON to stdout. Read and parse the output directly from the Bash tool result.

## Workflow

### Step 1: Identify the target MR(s)

If the user provides a specific MR URL or IID, use that. Otherwise, list open MRs:

```bash
glab mr list --author=@me
```

If no open MRs, tell the user and stop.

### Step 2: Run the script

```bash
python3 ~/.claude/skills/assign-mr-reviewers/scripts/assign_mr_reviewers.py <MR_URL>
```

or:

```bash
python3 ~/.claude/skills/assign-mr-reviewers/scripts/assign_mr_reviewers.py <PROJECT_PATH> <MR_IID>
```

With exclusions (e.g. people who are away):

```bash
python3 ~/.claude/skills/assign-mr-reviewers/scripts/assign_mr_reviewers.py <MR_URL> --exclude nhughes,bsmith
```

The script does everything in one step:
1. Fetches MR details (author, reviewers, changed files, required approvals)
2. Checks CODEOWNERS, falls back to team-spectra group members
3. Picks reviewers (excluding author, existing reviewers, and any `--exclude` users)
4. Assigns them via `glab mr update`

It outputs JSON with: status, project_path, mr_iid, title, author, existing_reviewers, required_approvals, assigned, source, remaining_candidates.

If `status` is `already_has_reviewers`, the MR already has enough reviewers — tell the user and stop (unless they asked to reassign).

### Step 3: Report

Show a summary:

```
MR !421 (fix: lodash fixes) → added reviewer: @rtseng (team-spectra fallback)
```

Indicate whether reviewers came from CODEOWNERS or the team-spectra fallback.

## Edge cases

- **MR already has enough reviewers**: Skip and mention in summary, unless user asks to reassign.
- **Not enough eligible members**: Assign as many as possible and warn the user.
- **Multiple CODEOWNERS matches**: Collect all unique owners across changed files. If more owners than required approvals, assign all of them.
- **CODEOWNERS specifies a group**: Resolve group members via the API and pick from them.
