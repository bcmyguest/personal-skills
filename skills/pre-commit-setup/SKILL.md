---
name: pre-commit-setup
description: Stand up pre-commit in a repository from shipped config fragments — the standard hygiene hooks (whitespace, EOF newline, YAML/JSON/TOML checks, large-file and merge-conflict guards) plus an optional Conventional Commits layer — then install the git hooks and run across the repo. Also the canonical policy for updating hooks safely (autoupdate --freeze to immutable SHAs). Use when adding pre-commit to a project, asked to "set up pre-commit" or "add commit hooks", when updating/refreshing hook pins, or when called from a project-setup skill (rust-setup, uv-setup, react-ts-setup) that layers its language hooks on top.
---

# Setting up pre-commit

This skill owns the shared pre-commit machinery: the **baseline hygiene
hooks**, the optional **Conventional Commits** layer, hook installation, and
the **update policy**. Language-specific hooks (cargo fmt/clippy, ruff +
pyrefly, …) are shipped as fragments by the relevant project-setup skill and
concatenated with the fragments here — nothing is retyped.

## 1. Install pre-commit

Use the project's package manager so the dependency is tracked. In a uv
project:

```bash
uv add --dev pre-commit
```

If there's no package manager to track it (e.g. a Rust or plain repo), **ask
the user** how to install it (`pipx install pre-commit`,
`brew install pre-commit`, system package) rather than picking for them.
(Prefix the commands below with `uv run` when it's a tracked dev dep.)

## 2. Assemble `.pre-commit-config.yaml` from fragments

The config is built by concatenating fragment files under one `repos:`
header. Fragments shipped here in [`templates/`](templates/):

- `hygiene.repos.yaml` — the baseline every repo gets.
- `conventional-commits.repos.yaml` — commit-msg enforcement; include it
  whenever release automation derives versions from commit types (the
  rust-setup / uv-setup / release-setup flows).

A calling setup skill contributes its own language fragment (e.g.
`rust-setup/templates/pre-commit-rust.repos.yaml`,
`uv-setup/templates/pre-commit-python.repos.yaml`). With `$PCS` = this
skill's directory:

```bash
{ echo 'repos:'; cat "$PCS/templates/hygiene.repos.yaml" \
    "$PCS/templates/conventional-commits.repos.yaml" \
    "<language fragment, if any>"; } > .pre-commit-config.yaml
```

Drop the conventional-commits line if the repo has no commit-type-driven
release automation. If `.pre-commit-config.yaml` already exists, **merge**
the fragments' `repos:` entries in rather than overwriting other hooks.

## 3. Install the git hooks and run

```bash
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
pre-commit run --all-files
```

(The `--hook-type commit-msg` matters only when the conventional-commits
fragment is included, and is harmless otherwise.) The first `run --all-files`
will likely modify files (whitespace, newlines, line endings) — review, then
**commit the fixes** so future diffs stay clean.

## 4. Updating hooks — the canonical policy

Pre-commit hooks run arbitrary third-party code on every commit, so treat
updates as a supply-chain surface. This section is the single home of that
policy — setup skills point here instead of restating it:

- **Refresh pins with `pre-commit autoupdate --freeze`** — both right after
  copying fragments (their pins were current when written, not today) and on
  later deliberate updates. `--freeze` resolves each hook to the immutable
  40-char **commit SHA**, keeping the tag as a comment.
- **Never plain `autoupdate`.** It moves `rev` to the latest *tag*, and git
  tags are mutable — a compromised maintainer can re-point one at malicious
  code.
- **Review every bump before committing** — compare old SHA to new upstream,
  confirm the release comes from the expected maintainer. Update
  deliberately, not on a schedule.
- Local (`repo: local`) hooks have no pin — they run the project's own
  toolchain and update with it.
- In CI, `pre-commit run --all-files` enforces the same checks on every push
  (the rust-setup/uv-setup CI templates instead run the underlying tools
  directly — same effect, pick one, not both).

## Checklist

- [ ] pre-commit installed (tracked as a dev dependency where possible)
- [ ] config assembled from fragments (hygiene + conventional-commits if release automation + language layer from the calling skill)
- [ ] `pre-commit autoupdate --freeze` run after assembly
- [ ] hooks installed (both hook types when conventional-commits is included); `run --all-files` passes
- [ ] formatting fixes from the first run committed
