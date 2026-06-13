# personal-skills

A local [Claude Code](https://docs.claude.com/en/docs/claude-code) plugin marketplace —
my personal toolkit. All skills live in a single [`skills/`](skills) pool; plugins are
thematic groupings defined in [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json),
each selecting a subset of those skills (the same pattern as
[anthropics/skills](https://github.com/anthropics/skills)).

## Plugins

| Plugin | Skills | Docs |
|--------|--------|------|
| `python-dev` | uv-setup, uv-develop, pre-commit-setup | [docs/python-dev.md](docs/python-dev.md) |
| `git-tools` | git-attribution (+ enforcing hook) | [docs/git-tools.md](docs/git-tools.md) |
| `agent-workflow` | handoff, watchlist | [docs/agent-workflow.md](docs/agent-workflow.md) |
| `frontend` | senior-frontend-refactor | [docs/frontend.md](docs/frontend.md) |
| `ansible` | add-ansible-role | [docs/ansible.md](docs/ansible.md) |
| `lemonade` | debug-lemonade | [docs/lemonade.md](docs/lemonade.md) |

## Layout

```
.claude-plugin/marketplace.json   # plugins = groupings over the shared skills/ pool
plugins.json                      # marketplaces + enable list for install-plugins.sh
install-plugins.sh                # idempotent installer
skills/                           # one dir per skill (shared by the source:"./" plugins)
  uv-setup/  uv-develop/  pre-commit-setup/
  handoff/   watchlist/   senior-frontend-refactor/
  add-ansible-role/  debug-lemonade/
plugins/git-tools/                # self-contained plugin (isolated hook)
  .claude-plugin/plugin.json
  hooks/hooks.json
  scripts/attribution-guard.py
  skills/git-attribution/
docs/                             # one doc per plugin
```

Most plugins use `source: "./"` and list their skills explicitly, so they share the
root `skills/` pool. `git-tools` is the exception: it ships a `PreToolUse` hook, and a
hook at the repo root would attach to *every* `source: "./"` plugin — so it lives in its
own directory (`plugins/git-tools`) to keep the hook isolated to that plugin.

## Install on a new machine

```bash
git clone git@github.com:bcmyguest/personal-skills.git ~/personal-skills
~/personal-skills/install-plugins.sh
```

The installer registers this directory as a local marketplace, plus the
`anthropic-agent-skills` and `caveman` GitHub marketplaces, and enables the plugins
listed in [`plugins.json`](plugins.json).

## License

See [LICENSE](LICENSE). All rights reserved; in particular, the contents may **not** be
used as training, fine-tuning, or evaluation data for machine-learning or AI systems.
