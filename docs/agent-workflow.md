# agent-workflow

Helpers for long agent sessions and tracking upstream work.

| Skill | What it does |
|-------|--------------|
| `handoff` | Write a concise handoff summary of the current troubleshooting/debugging activity so the next agent or session can continue without re-deriving anything. |
| `watchlist` | Check upstream GitHub issues, PRs, and releases you're waiting on, and report what moved since the last check. |

Skills live in [`../skills/`](../skills); this plugin selects them via `skills` in the
[marketplace manifest](../.claude-plugin/marketplace.json).
