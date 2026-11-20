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

Without installing, point an agent at the file: "Follow `skills/diagnose-failure/SKILL.md`
from bcmyguest/personal-skills."

## Skills

| Skill | What |
|-------|------|
| `diagnose-failure` | Find and fix the actual cause of a failure, with fast and strict paths |
| `sanity-check-numbers` | Decide whether a metric, score, or cost is real before trusting it |
| `audit-agent-diff` | Check an agent-written diff for overreach before it lands |
| `land-change` | Commit, push, PR, and merge exactly the intended change |
| `port-changes` | Bring selected work across branches, PRs, or archives |
| `match-ui-to-reference` | Converge a UI on a design through numbered deltas and screenshots |
| `resolve-design-decisions` | Settle costly-to-reverse choices in dependency-ordered question rounds |
| `research-options` | Compare options with primary sources and spikes |
| `independent-review` | Get a verdict from a reviewer that did not write the artifact |
| `brief-agent` | Write a self-contained brief for the next agent or session |
| `sorry-what` | Re-explain the last response, or items `2,4` of it, as context, a code snippet, a diagram, or an overview |
| `show-me` | Point to the code or source behind the last response: snippet plus a permalink on GitHub, GitLab, Azure DevOps, Bitbucket, or Gitea, or the doc section |
| `to-diagram` | Draw the subject as the diagram type that fits, checked by Mermaid's parser, inline where the chat renders it and as an HTML file elsewhere |
| `plan-experiment` | Pre-register a comparison before spending on it |
| `run-experiment` | Run an approved plan, monitor it, and keep it resumable |
| `write-evidence-report` | Publish results that say exactly what the evidence supports |
| `uv-setup` | Scaffold a Python repo with uv from shipped templates — CI, auto-releases, PyPI Trusted Publishing |
| `uv-develop` | Corrective uv operations — targeted upgrades, `--locked` vs `--frozen`, venv repair |
| `pre-commit-setup` | Stand up pre-commit from shipped fragments — hygiene + Conventional Commits + hook-update policy |
| `release-setup` | Release automation from Conventional Commits — canonical git-cliff config + GitLab tag/release pipeline |
| `use-template` | Start a repo from your GitHub template repos (Rust CLI, npm, Python, docs site, skills pack) |
| `watchlist` | Track upstream GitHub issues, PRs, and releases |
| `assign-mr-reviewers` | Assign CODEOWNERS reviewers to your open GitLab MRs |
| `babysit-pr` | Work an open PR/MR on GitHub, GitLab, or Azure DevOps — batch small review fixes, round up design questions, fix failing jobs, restack |
| `react-ts-setup` | Scaffold a React + TypeScript repo with Vite + full toolchain |
| `rust-setup` | Scaffold a Rust repo from shipped templates — CI, auto-releases, crates.io Trusted Publishing |
| `senior-frontend-refactor` | Refactor frontend code as a senior engineer |
| `add-ansible-role` | Add a tool-install role to a personal Ansible playbook |
| `debug-lemonade` | Diagnose lemonade-server on the Strix Halo box |
| `security-audit` | Find and adversarially verify a codebase's 1–3 worst security vulnerabilities (capped Sonnet multi-agent workflow) |
| `efficiency-audit` | Find and verify a codebase's 1–3 worst efficiency problems with concrete cost analyses (capped Sonnet multi-agent workflow) |

Layout: one folder per skill under [`skills/`](skills), each with a `SKILL.md` plus any
resources it needs (`references/`, `templates/`, `config.json`).

`writing-with-clarity` moved to [bcmyguest/claude-memory](https://github.com/bcmyguest/claude-memory);
`fast-hypothesis` and `reproduce-bug` are folded into `diagnose-failure`.

## wteam teams

[`wteam/`](wteam) mirrors the [wteam](https://github.com/bcmyguest/wteam) config dir: one
`team.dot` per team in [`wteam/teams/`](wteam/teams), where each step names its skill with
`skill=`, plus the `artifact-review` phase and its persona roles. Import it:

```bash
scripts/install-wteam.sh   # copies wteam/ into ${WTEAM_CONFIG_DIR:-~/.config/wteam}, then lists teams and phases
wteam run --pairs 1 --team experiment --slug <slug>
```

Hand-offs outside a team are each skill's `## Next` section.

The portable `handoff` and `pickup` skills moved to their own repo —
[bcmyguest/baton](https://github.com/bcmyguest/baton) (`npx skills add bcmyguest/baton`).

## Claude Code plugins (hook-bearing — not plain skills)

Some tools need a hook, which the portable skills format can't carry. Those are Claude
Code plugins and now live in their own repos:

| Plugin | What | Repo |
|--------|------|------|
| `assisted-by` | `git-attribution` skill **plus** a PreToolUse hook that enforces AI commit attribution on `git commit` | [bcmyguest/assisted-by](https://github.com/bcmyguest/assisted-by) |
| `skill-inject` | Local, model-agnostic automatic skill injection | [bcmyguest/skill-injector](https://github.com/bcmyguest/skill-injector) |
| `claude-memory` | Session conventions (MEMORY.md) plus `/install` script for `@include` injection | [bcmyguest/claude-memory](https://github.com/bcmyguest/claude-memory) |

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
  uv-setup/                  SKILL.md + templates/
  uv-develop/                SKILL.md
  pre-commit-setup/          SKILL.md + templates/
  release-setup/             SKILL.md + templates/
  use-template/              SKILL.md
  watchlist/                 SKILL.md
  assign-mr-reviewers/       SKILL.md + scripts/
  babysit-pr/                SKILL.md + FORGES.md
  react-ts-setup/            SKILL.md + references/
  rust-setup/                SKILL.md + templates/
  senior-frontend-refactor/  SKILL.md
  add-ansible-role/          SKILL.md + config.json + templates/
  debug-lemonade/            SKILL.md
  diagnose-failure/ … write-evidence-report/   SKILL.md each (method skills, see DECISIONS.md)
  sorry-what/                SKILL.md
  show-me/                   SKILL.md + scripts/permalink.py (+ test_permalink.py)
  to-diagram/                SKILL.md + scripts/diagram.mjs (+ test_diagram.mjs) + references/
  security-audit/            SKILL.md + scripts/ + evals/
  efficiency-audit/          SKILL.md + scripts/ + evals/
  _fixtures/                 shared eval fixture (vuln-app) for the audit skills
skills.sh.json                   # skills.sh groupings for the skills above
plugins.json                     # external marketplaces + enable list for install-plugins.sh
install-plugins.sh               # personal-machine bootstrap (hook-bearing plugins)
IDEAS.md                         # parked skill candidates and config snippets
wteam/teams/<name>/team.dot      # wteam teams, one per workflow
wteam/phases/, wteam/roles/      # the artifact-review phase and its personas
scripts/install-wteam.sh         # copies wteam/ into the wteam config dir
DECISIONS.md                     # why the method skills look the way they do
```

## Tests

```bash
python3 skills/show-me/scripts/test_permalink.py           # forge parsing, link shapes, no-link reasons
node --test skills/to-diagram/scripts/test_diagram.mjs     # Mermaid parse errors, node cap, HTML output
```

## License

See [LICENSE](LICENSE). All rights reserved; in particular, the contents may **not** be
used as training, fine-tuning, or evaluation data for machine-learning or AI systems.
