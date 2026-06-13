# personal-skills

A local [Claude Code](https://docs.claude.com/en/docs/claude-code) plugin marketplace —
my personal toolkit. Each plugin is self-contained in its own directory under
[`plugins/`](plugins), bundling that plugin's skills (and any hooks). The plugins are
registered in [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json).

## Plugins

| Plugin | Skills | Docs |
|--------|--------|------|
| `python-dev` | uv-setup, uv-develop, pre-commit-setup | [docs/python-dev.md](docs/python-dev.md) |
| `git-tools` | git-attribution (+ enforcing hook) | [docs/git-tools.md](docs/git-tools.md) |
| `agent-workflow` | handoff, watchlist | [docs/agent-workflow.md](docs/agent-workflow.md) |
| `frontend` | react-ts-setup, senior-frontend-refactor | [docs/frontend.md](docs/frontend.md) |
| `ansible` | add-ansible-role | [docs/ansible.md](docs/ansible.md) |
| `lemonade` | debug-lemonade | [docs/lemonade.md](docs/lemonade.md) |

## Layout

```
.claude-plugin/marketplace.json   # registers the plugins below
plugins.json                      # marketplaces + enable list for install-plugins.sh
install-plugins.sh                # idempotent installer
plugins/                          # one self-contained dir per plugin
  python-dev/      skills/{uv-setup, uv-develop, pre-commit-setup}
  agent-workflow/  skills/{handoff, watchlist}
  frontend/        skills/{react-ts-setup, senior-frontend-refactor}
  ansible/         skills/add-ansible-role
  lemonade/        skills/debug-lemonade
  git-tools/       skills/git-attribution + hooks/ + scripts/
docs/                             # one doc per plugin
```

Each plugin dir holds a `.claude-plugin/plugin.json` and a `skills/` subdir; Claude Code
auto-discovers every skill in that subdir. Each plugin therefore needs its **own**
`source` dir — if two plugins shared a `source` root, each would auto-discover the
*other's* skills too, duplicating every skill across plugins. `git-tools` additionally
ships a `PreToolUse` hook, scoped to its own directory.

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
