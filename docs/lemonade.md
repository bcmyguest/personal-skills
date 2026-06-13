# lemonade

Machine-specific runbook for the Strix Halo box.

| Skill | What it does |
|-------|--------------|
| `debug-lemonade` | Diagnose lemonade-server (lemond) problems — server not responding, models missing from Open WebUI, a model failing to load, slow inference, or ROCm/Vulkan/GPU issues. |

Skills live in [`../skills/`](../skills); this plugin selects them via `skills` in the
[marketplace manifest](../.claude-plugin/marketplace.json).
