---
name: use-template
description: Start a new repo from one of the user's GitHub template repos (bcmyguest/template-*) — Rust CLI, npm package, uv Python package, docs site, or agent skills pack — each already wired with CI, pre-commit + Conventional Commits, Dependabot, pinned actions and a two-step git-cliff release. Use when the user asks to start, create or bootstrap a new project, package, CLI, docs site or skills pack. Prefer it over rust-setup and uv-setup unless the user asks for those by name.
---

# use-template

Each template is a working, CI-green repo. A new project is a copy plus one
rename script.

## Templates

| Kind | Template | Ships |
|---|---|---|
| Rust CLI + library | `template-rust-cli` | static binaries for 5 targets, `install.sh`/`install.ps1`, crates.io when `PUBLISH_CRATES=true` |
| npm package | `template-node-package` | ESM, `node:test`, pack check, npm when `PUBLISH_NPM=true` |
| Python package | `template-python-package` | uv, ruff, pyrefly, wheel smoke test, PyPI when `PUBLISH_PYPI=true` |
| Docs site | `template-docs-site` | `docs/*.md` → Astro + Starlight, link and contrast checks, `llms.txt`, GitHub Pages |
| Skills pack | `template-skills-pack` | `skills/*/SKILL.md`, Claude plugin + marketplace, `skills.sh.json`, frontmatter check |

The live list is the source of truth; it catches templates added after this table:

```bash
gh repo list bcmyguest --json name,isTemplate,description --jq '.[] | select(.isTemplate) | "\(.name): \(.description)"'
```

Ask the user which kind when the request does not settle it. A docs site for an
existing repo: copy `site/`, `docs/` and `.github/workflows/pages.yml` from the
template instead of creating a repo.

## Steps

1. Name and one-line description from the user. Name: lowercase-hyphenated.
   Visibility defaults to private; confirm before creating a public repo.
2. Create and clone:
   ```bash
   gh repo create NAME --template bcmyguest/TEMPLATE --private --clone
   cd NAME && git pull   # the first clone can race the template copy and come back empty
   ```
3. Rename:
   ```bash
   scripts/init.sh NAME "DESCRIPTION"
   ```
   It replaces the template name in contents and paths, fills the description,
   takes the owner from `origin`, strips the README's template section and
   deletes itself. Done when `scripts/init.sh` is gone.
4. Run the template's checks (listed under **Commands** in its `CLAUDE.md`);
   all pass before the first commit.
5. Commit `chore: init from TEMPLATE`, push, and install hooks:
   ```bash
   pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
   ```
6. Done when the CI run on `main` is green (`gh run watch`). The `placeholders`
   job fails while `scripts/init.sh` still exists.

## After setup

Releases: run the **Prepare release** workflow (`dry_run` first); never bump
versions by hand. Registry publishing stays off until the user sets up trusted
publishing and the repo variable named in the README; walk them through it
only when they ask to publish.

## Gotchas

- Raw-URL installers (`curl … | sh`) and `npx skills add` need a public repo.
- GitHub Pages needs a public repo on this account, and a one-time enable with
  the user's token: `gh api -X POST repos/OWNER/REPO/pages -f build_type=workflow`.
- git-cliff run locally against a repo not yet on GitHub fails on a 404
  contributor lookup; use `git-cliff --offline`.
