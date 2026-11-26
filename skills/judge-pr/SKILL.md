---
name: judge-pr
argument-hint: "[pr-link | pr-number] [babysit]"
description: >-
  One-shot judgment of a PR or MR by a fresh strong-model subagent: a verdict,
  findings split into "just change it" items for bulk approval and judgment-call
  items to agree or disagree with, and applyable patches, shown with sketch-it
  views. Default mode posts nothing on the PR; a babysit mode posts the findings
  as review comments then hands off to babysit-pr. Use when the user asks to
  judge, get a second opinion on, or one-shot review a PR.
---
# Judge a PR

One pass. A fresh reviewer reads the PR cold and comes back with judgment you can act on in one sitting. `babysit-pr` watches and fixes for hours; `judge-pr` rules once. The default posts nothing on the PR.

## 1. Find the PR

The argument names the PR, as in `babysit-pr`:

- **A link**: the host picks the forge, the path gives the repository and the number. Work in a local checkout of that repository; when the current directory is another repository, find the checkout or ask where it is.
- **A number**: that PR or MR in the current repository.
- **Nothing**: the open PR whose head is the current branch. With no open PR for the branch, say so and stop.

A bare `babysit` (or "and babysit") anywhere in the argument selects **babysit mode**; without it the mode is **feedback**.

Fetch the diff and check out the head branch (use a separate worktree with uncommitted work in the tree):

| Forge | Diff | Checkout |
| --- | --- | --- |
| GitHub (`gh`) | `gh pr diff <n>` | `gh pr checkout <n>` |
| GitLab (`glab`) | `glab mr diff <n>` | `glab mr checkout <n>` |
| Azure DevOps (`az`) | `az repos pr diff` per `az repos pr --help` | branch fetch and switch |

## 2. Run the judge on a subagent

Spawn a fresh subagent — never judge in this session, you have context that biases you. Pick a strong reasoning model, e.g. Claude Opus or another provider's strong tier; the strongest model available is not needed, the cheap or fast default is not enough. When the host cannot choose a model, use its default and say so in the report.

Give the subagent, and nothing else from this conversation:

- The repository path, the base branch, and the full diff. It may read the surrounding code, tests, and repo docs; it is read-only.
- The task, in these terms:

  > Review this PR. Return:
  >
  > 1. **Verdict** — ship, fix-then-ship, or blocked, with the reason in one paragraph.
  > 2. **Just change it** — findings that should change regardless of taste: bugs, security holes, broken or missing tests, dead code, violations of a convention the repo states. One to two sentences each: what is wrong, what it costs, at `file:line`. Attach a patch to each.
  > 3. **Judgment calls** — design choices, style preferences, scope, and anything you are unsure about. One short paragraph each, with the alternative, and what disagreeing costs. Attach a patch when you can write one concretely.
  > 4. **Checked and clean** — the three to five things you looked hardest at and found sound, one line each.
  >
  > Each patch is a unified diff that applies to the head branch with `git apply`, one file per finding, nothing else touched. Judge the code, not the author; flag a finding only with a concrete consequence.

## 3. Verify before showing

You are not a courier. Before presenting anything:

- Check every finding against the code. Drop ones that do not hold; mark ones you cannot confirm as unverified rather than deleting them silently.
- Save each surviving patch to `${TMPDIR:-/tmp}/judge-pr/<pr-slug>/NN-<slug>.patch`, numbered, `git apply`-clean against the checked-out head (test-apply each with `git apply --check`).
- Where a finding is about structure — flow, module boundaries, call order — render the shape with the `sketch-it` skill (a sequence diagram usually; a page when regions or before/after must be compared) so the judgment is visible, not just described.

## 4. Report

Present, in this order:

1. **Verdict** — the paragraph, plus model and what the judge read.
2. **Just change it** — a numbered list; each item is the finding in one or two sentences, `file:line`, and the patch path. End the block with the bulk ask: *"Reply `all` to apply every patch and run the tests, or the numbers to skip."*
3. **Judgment calls** — a numbered list; each item states the question, the judge's lean, the alternative, and the patch path if any. Ask the user to answer each **agree** / **disagree**, or just the ones they care about.
4. **Checked and clean** — the one-liners.

Nothing is posted on the PR in this step, in either mode.

## 5. Apply what was approved

- On `all` or a list of numbers, apply those patches on the PR branch, run the project's tests, lint, and type check, and report what passed. Push only when the user says to.
- Answer each judgment call the user marks agree with its patch the same way; on disagree, record the decision in one line and move on.

## 6. Modes

- **feedback** (default): stop after step 5. Post no comment, review, or reply anywhere on the PR.
- **babysit**: after step 4 — before waiting for answers — post the findings on the PR: one inline comment per finding at its `file:line` with the patch in a fenced `diff` block, and one top-level comment with the verdict. Mark each comment's body `judge-pr:` and keep items one to two sentences. Then run the `babysit-pr` skill on the PR and let it work through its own rules from there.
