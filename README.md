# personal-skills

Portable [agent skills](https://skills.sh) — one `SKILL.md` per skill,
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
| `pickup` | Resume from the latest handoff — read, confirm, then continue |
| `watchlist` | Track upstream GitHub issues, PRs, and releases |
| `react-ts-setup` | Scaffold a React + TypeScript repo with Vite + full toolchain |
| `senior-frontend-refactor` | Refactor frontend code as a senior engineer |
| `add-ansible-role` | Add a tool-install role to a personal Ansible playbook |
| `debug-lemonade` | Diagnose lemonade-server on the Strix Halo box |

Layout: one folder per skill under [`skills/`](skills), each with a `SKILL.md` plus any
resources it needs (`references/`, `templates/`, `config.json`).

## Claude Code plugins (hook-bearing — not plain skills)

Some tools need a hook, which the portable skills format can't carry. Those are Claude
Code plugins and now live in their own repos:

| Plugin | What | Repo |
|--------|------|------|
| `assisted-by` | `git-attribution` skill **plus** a PreToolUse hook that enforces AI commit attribution on `git commit` | [bcmyguest/assisted-by](https://github.com/bcmyguest/assisted-by) |
| `skill-inject` | Local, model-agnostic automatic skill injection | [bcmyguest/skill-injector](https://github.com/bcmyguest/skill-injector) |

```bash
claude plugin marketplace add bcmyguest/assisted-by
claude plugin install assisted-by@assisted-by
```

## Bootstrap my own machine

For my own setup, [`install-plugins.sh`](install-plugins.sh) is an idempotent
bootstrap that registers the `assisted-by`, `caveman`, and `anthropic-agent-skills`
GitHub marketplaces, then installs and enables the plugins listed in
[`plugins.json`](plugins.json). It does **not** install the portable `skills/` — use
`npx skills` above for those.

```bash
git clone git@github.com:bcmyguest/personal-skills.git ~/personal-skills
~/personal-skills/install-plugins.sh
```

## Layout

Two tracks: portable skills install with `npx skills`; hook-bearing plugins install
through the Claude Code marketplace.

```
skills/                          # portable agent skills — one folder per skill (npx skills)
  uv-setup/                  SKILL.md
  uv-develop/                SKILL.md
  pre-commit-setup/          SKILL.md
  handoff/                   SKILL.md
  pickup/                    SKILL.md
  watchlist/                 SKILL.md
  react-ts-setup/            SKILL.md + references/
  senior-frontend-refactor/  SKILL.md
  add-ansible-role/          SKILL.md + config.json + templates/
  debug-lemonade/            SKILL.md
skills.sh.json                   # skills.sh groupings for the skills above
plugins.json                     # external marketplaces + enable list for install-plugins.sh
install-plugins.sh               # personal-machine bootstrap (hook-bearing plugins)
```

## License

See [LICENSE](LICENSE). All rights reserved; in particular, the contents may **not** be
used as training, fine-tuning, or evaluation data for machine-learning or AI systems.
