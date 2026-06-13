# git-tools

Git commit hygiene following the Linux-kernel coding-assistants policy.

| Skill | What it does |
|-------|--------------|
| `git-attribution` | Attribute AI assistance with an `Assisted-by:` trailer; never `Signed-off-by:` or `Co-Authored-By: Claude` from the AI. |

## Hook

Unlike the other plugins (which share the repo root via `source: "./"`), `git-tools`
is a self-contained plugin directory at
[`../plugins/git-tools`](../plugins/git-tools) so its hook stays isolated. It ships a
`PreToolUse` hook (`scripts/attribution-guard.py`, wired in `hooks/hooks.json`) that
inspects `git commit` commands and blocks any that author a message without an
`Assisted-by:` trailer, or that try to add an AI `Signed-off-by` / `Co-Authored-By: Claude`.
