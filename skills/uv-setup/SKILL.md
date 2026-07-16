---
name: uv-setup
description: Scaffold a new Python project with uv by copying shipped template files — uv init --package, then overlay templates for ruff + pyrefly (local hook) pre-commit, Conventional Commits, CI entirely through uv, Dependabot (uv + actions), git-cliff auto-release on merge, PyPI Trusted Publishing (pending publisher, never a token), and a multi-stage Dockerfile for services. Use when starting a new Python project, scaffolding a repo, or asked to "set up" or "initialize" a Python project. For day-to-day work in an existing project, use uv-develop instead.
---

# Scaffolding a uv project

Everything ships as **real files under [`templates/`](templates/)** — copy
them verbatim, run one substitution pass, make the few marked edits. Do
**not** retype configs from memory or improvise structure: the templates are
the source of truth, and the decisions below only change *which* files get
copied. `uv init` still runs first — uv owns the generated things (git repo,
`src/` dir, `uv.lock`, dev-dep versions); the templates overlay everything
opinionated on top.

What the result looks like: a packaged `src/` layout with a tested
hello-world mock, CI running lint + format + types + tests all through uv,
hygiene + Conventional Commits enforced at commit time, Dependabot keeping
uv.lock and actions current, and — if chosen — every merge to `main`
auto-releasing via git-cliff, with PyPI uploads via Trusted Publishing only.
Day-to-day work afterwards follows the **uv-develop** skill.

## 0. Ask the decisions

| Decision             | Options                                              | Default        |
| -------------------- | ---------------------------------------------------- | -------------- |
| Project name         | kebab-case; must be free on PyPI if publishing       | ask            |
| GitHub repo slug     | `owner/repo`                                         | ask            |
| One-line description | —                                                    | ask            |
| Project type         | library · application/CLI · service · script         | ask            |
| Python version       | recent stable (avoid pre-releases)                   | **3.13**       |
| License              | SPDX expression (`MIT`, `Apache-2.0`, …)             | **MIT**        |
| Release automation   | auto-release every merge to `main` · none            | **auto-release** |
| Publish to PyPI      | yes (**Trusted Publishing only**) · no               | libraries: ask; others: usually no |

**Script / throwaway** (one-off, not importable) is the only non-templated
case: bare `uv init` (no `--package`), none of this skill applies — stop here.

## 1. Generate, then copy the templates

```bash
uv init --package <project-name> && cd <project-name>
```

`$SKILL` below is this skill's directory; `<pkg>` is the package dir uv
created under `src/` (project name with `-` → `_`). Later copies overlay
earlier ones — the template pyproject.toml, README.md, and .gitignore
*replace* the generated ones on purpose.

```bash
# Always — the base layer (note the /. so dotfiles come along):
cp -R "$SKILL/templates/base/." .
cp -R "$SKILL/templates/package/." "src/<pkg>/"   # mock module + py.typed

# Project type — exactly one pyproject.toml:
cp "$SKILL/templates/lib/pyproject.toml" .        # library (no console script)
cp "$SKILL/templates/app/pyproject.toml" .        # application/CLI AND service

# Service only — in addition to app/pyproject.toml:
cp "$SKILL/templates/service/Dockerfile" "$SKILL/templates/service/.dockerignore" .

# Release automation (skip both lines entirely if "none"):
cp "$SKILL/templates/release/cliff.toml" .
cp "$SKILL/templates/release/<variant>" .github/workflows/release.yml
```

Release variants — pick one: `release-pypi.yml` if publishing to PyPI
(tag + GitHub release + OIDC upload), else `release-github.yml` (tag +
GitHub release only).

## 2. Fill the placeholders

Six placeholders, one pass (adjust values; `{{package_snake}}` = project
name with `-` → `_`):

```bash
grep -rlF '{{' --exclude-dir=.venv --exclude-dir=.git . | xargs sed -i \
  -e 's/{{project_name}}/my-tool/g' \
  -e 's/{{package_snake}}/my_tool/g' \
  -e 's|{{repo_slug}}|owner/my-tool|g' \
  -e 's/{{description}}/One-line description/g' \
  -e 's|{{license}}|MIT|g' \
  -e 's/{{python_version}}/3.13/g'
```

(macOS/BSD sed: `sed -i ''`.) Afterwards `grep -rF '{{' .` must only hit
GitHub-Actions `${{ … }}` expressions and cliff.toml's Tera template.

## 3. Pin Python, add the dev toolchain, lock

Do this **after** the placeholder pass (uv can't parse a pyproject with
`{{…}}` in it). Versions come from today, never from templates:

```bash
uv python pin 3.13                            # writes .python-version — commit it
uv add --dev pytest ruff pyrefly pre-commit   # scaffold-time versions into uv.lock
uv sync
```

Keep `.python-version`, `requires-python`, `tool.pyrefly.python-version` —
and for services both Dockerfile `FROM` lines — on the **same** version;
bump them together (the Dockerfile explains why the venv breaks otherwise).

## 4. LICENSE and marked edits

- Write the `LICENSE` file for the chosen SPDX expression (canonical text,
  copyright line = current year + author).
- Work through every `<!-- DELETE … -->` / `<!-- KEEP … -->` marker in
  `README.md` and `CLAUDE.md` — they mark the branch-dependent blocks
  (badges, install methods, Build & validate, Releases wording).
- Verify the three `.gitignore` invariants survived (the shipped file
  documents them: `.python-version` committed; `.venv/` and `__pycache__/`
  ignored at any depth, no leading slash):

```bash
git check-ignore -v .python-version    # expect NO match (exit 1) — it's committed
git check-ignore .venv a/b/.venv a/__pycache__/x.pyc   # expect all three printed
```

## 5. Refresh the moving parts — don't trust shipped pins

- `uv run pre-commit autoupdate --freeze` — moves each hook to the latest
  release's **commit SHA** (immutable; the tag stays as a comment). The
  supply-chain rationale lives in the **pre-commit-setup** skill; that skill
  also owns the baseline hygiene hooks already included in the shipped
  config.
- Check the `uses:` pins in the copied workflows against upstream latest
  (`setup-uv` publishes no moving major tags since v8 — pin full versions).
  After the first push, **Dependabot owns this** (`.github/dependabot.yml`
  covers the `uv` ecosystem — pyproject.toml + uv.lock — and
  `github-actions`, weekly, grouped).

## 6. Verify green, then first commit

```bash
uv run ruff check && uv run ruff format --check
uv run pyrefly check
uv run -m pytest
uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
uv run pre-commit run --all-files
git add -A && git commit -m "feat: scaffold <project-name>"
```

The commit message **must** be conventional — git-cliff derives every version
bump from commit types, starting with this one (`feat:` → the initial
`v0.1.0`). The mock `greet` code exists so tests demonstrably pass; it gets
replaced by real code later. Libraries: also prove the wheel —

```bash
uv build
uv run --isolated --no-project --with dist/<pkg>-*.whl \
  python -c "from <pkg> import greet; print(greet('wheel'))"
```

Services: `docker build -t <project-name> . && docker run --rm <project-name>`.

## 7. Publishing to PyPI — Trusted Publishing ONLY

Never store a PyPI token in GitHub secrets: `release.yml` exchanges the
workflow run's OIDC identity for a short-lived token via
`pypa/gh-action-pypi-publish` (that's the `id-token: write` permission and
the `environment: pypi`). Unlike crates.io, **PyPI supports pending
publishers** — the trusted publisher is registered *before* the project's
first upload, so no token is ever needed:

1. **Register the pending publisher:** PyPI → account → *Publishing* → *Add
   a new pending publisher* (GitHub): project name, repository owner + name,
   workflow filename `release.yml`, environment `pypi` (matches the
   `environment:` in the workflow's publish job).
2. **Push to GitHub.** The first release run tags `v0.1.0`, publishes the
   GitHub release, and the OIDC upload creates the PyPI project — the
   pending publisher becomes the project's normal trusted publisher.

If the publish job ever fails (e.g. publisher not registered yet), the tag
and GitHub release have already landed — register the publisher and re-run
via `workflow_dispatch`, which republishes the latest tag idempotently.

## 8. After the push

- Confirm the `ci` and `release` runs are green and Dependabot is active
  (Insights → Dependency graph → Dependabot).
- The repo's `CLAUDE.md` ships as a skeleton — once real code replaces the
  mock, rewrite the description and add an Architecture section (`/init`
  drafts it); `AGENTS.md` just includes `CLAUDE.md`.

## Checklist

- [ ] decisions asked (name, slug, description, type, Python, license, releases, PyPI)
- [ ] `uv init --package` run (script → bare init, skill over); base + package + pyproject variant (+ service, + release) copied, nothing retyped
- [ ] placeholder pass done; `grep -rF '{{'` shows only `${{ }}` / Tera hits
- [ ] `uv python pin` + `uv add --dev pytest ruff pyrefly pre-commit` + `uv sync`; `.python-version` and `uv.lock` committed
- [ ] LICENSE written; all `<!-- DELETE/KEEP -->` markers resolved; `git check-ignore` invariants verified
- [ ] `pre-commit autoupdate --freeze` run; action pins checked
- [ ] ruff + pyrefly + pytest green; hooks installed (pre-commit **and** commit-msg); first commit conventional
- [ ] library: wheel built and validated in an isolated env
- [ ] service: docker image builds and runs (Python versions in lockstep)
- [ ] PyPI (if publishing): pending publisher registered *before* the push; first release lands via OIDC only
