# personal-skills

Brandon Guest's portable [agent skills](https://skills.sh) — one `SKILL.md` per skill,
in the open skills format that any modern coding agent reads (Claude Code, Codex,
opencode, Cursor, …). The repo itself is the package; GitHub is the registry.

## Install

```bash
npx skills add bcmyguest/personal-skills          # pick skills + agents interactively
npx skills add bcmyguest/personal-skills --all    # every skill -> every detected agent
npx skills add bcmyguest/personal-skills -a claude-code -a codex -a opencode
npx skills update                                  # pull later updates
```

`skills` symlinks each skill into your agent's native skills directory (single source of
truth, easy updates), or copies with `--copy` where symlinks aren't supported. See
[vercel-labs/skills](https://github.com/vercel-labs/skills).

## Skills

| Skill | What |
|-------|------|
| `uv-setup` | Bootstrap a new Python project with uv |
| `uv-develop` | Day-to-day uv dependency + test workflow |
| `pre-commit-setup` | Stand up pre-commit with the standard hygiene hooks |
| `handoff` | Write a troubleshooting/debugging handoff summary |
| `watchlist` | Track upstream GitHub issues, PRs, and releases |
| `react-ts-setup` | Scaffold a React + TypeScript repo with Vite + full toolchain |
| `senior-frontend-refactor` | Refactor frontend code as a senior engineer |
| `add-ansible-role` | Add a tool-install role to a personal Ansible playbook |
| `debug-lemonade` | Diagnose lemonade-server on the Strix Halo box |

Layout: one folder per skill under [`skills/`](skills), each with a `SKILL.md` plus any
resources it needs (`references/`, `templates/`, `config.json`).

## Claude Code plugins (hook-bearing — not plain skills)

Some tools need a hook, which the portable skills format can't carry. Those stay Claude
Code plugins under [`plugins/`](plugins), installed via the local marketplace:

```bash
claude plugin marketplace add bcmyguest/personal-skills
claude plugin install git-tools@personal-skills
```

| Plugin | What |
|--------|------|
| `git-tools` | `git-attribution` skill **plus** a PreToolUse hook that enforces AI commit attribution on `git commit` |

(`skill-inject` is migrating to its own repo and will ship like a normal binary.)

## License

See [LICENSE](LICENSE). All rights reserved; in particular, the contents may **not** be
used as training, fine-tuning, or evaluation data for machine-learning or AI systems.
