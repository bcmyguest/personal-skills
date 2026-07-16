---
name: release-setup
description: Automated releases from Conventional Commits with git-cliff — owns the canonical cliff.toml (commit types → semver bump → release notes) shared by every release flow in this repo, and fully implements the GitLab CI variant (tag vX.Y.Z + GitLab Release with generated notes on every default-branch push). Use when asked to automate releases, tagging, changelogs, or release notes — on GitLab directly; for GitHub the language setup skills (rust-setup, uv-setup) ship the workflow and copy cliff.toml from here.
---

# Release automation from Conventional Commits

One doctrine, shared by every release flow:

- Commit types drive the version: `feat` → minor, breaking → major, else
  patch; the first releasable commit produces `v0.1.0` (`initial_tag`).
- git-cliff computes the bump (`git cliff --bumped-version`) and renders the
  notes (`git cliff --unreleased --tag vX.Y.Z`); if the bumped version equals
  the latest tag, nothing releasable landed and the pipeline no-ops.
- The committed manifest version is never hand-bumped — release pipelines
  stamp the tag version in at build time.
- Registry publishing is **Trusted Publishing only** (OIDC) — never a
  long-lived token in CI secrets or variables.

[`templates/cliff.toml`](templates/cliff.toml) is the **canonical copy** of
the git-cliff config; the rust-setup and uv-setup release flows copy it from
here. Enforce the commit convention at commit time via the
**pre-commit-setup** skill's `conventional-commits.repos.yaml` fragment (with
the `--hook-type commit-msg` install).

## GitHub

The workflows are language-specific (binary matrices, crates.io vs PyPI), so
they ship with the language skill — follow **rust-setup** or **uv-setup**,
which copy `cliff.toml` from this skill.

## GitLab

Fully implemented here, language-agnostic, no placeholders:

```bash
cp "$SKILL/templates/cliff.toml" .
cp "$SKILL/templates/release.gitlab-ci.yml" .
```

Wire it into `.gitlab-ci.yml` — add the include, and make sure a `release`
stage exists (or rename the jobs' stage to one that does):

```yaml
stages: [..., release]

include:
  - local: release.gitlab-ci.yml
```

How it runs: on every push to the default branch, `release:plan` (git-cliff
image) computes the bumped version and renders `release-notes.md`;
`release:publish` (release-cli image) then creates the tag and the GitLab
Release with those notes.

### Caveats

- **Protected tags:** if the project protects `v*` tags
  (Settings → Repository → Protected tags), the job token can only create
  them when the pipeline runner is allowed to — grant Maintainers "create"
  on the pattern or relax it.
- **Full history:** the plan job sets `GIT_DEPTH: 0`; don't override it with
  a shallow clone setting, git-cliff needs the tags.
- **Publishing packages** is out of scope here, but both crates.io and PyPI
  support Trusted Publishing from GitLab CI (OIDC id_tokens); consult the
  registry's trusted publisher docs and append a publish job gated on
  `RELEASE == "true"`.

## Verify

- Locally, with git-cliff installed: `git cliff --bumped-version` prints the
  version the next default-branch push would release; `git cliff --unreleased
  --tag <that-version>` previews the notes.
- After pushing the wiring, CI → Editor → Validate (or the pipeline lint API)
  should accept the config, and the first pipeline on the default branch
  should show both jobs (no-op until a releasable commit lands).

## Checklist (GitLab)

- [ ] `cliff.toml` + `release.gitlab-ci.yml` copied verbatim to the repo root
- [ ] include added; `release` stage present in `stages:`
- [ ] Conventional Commits enforced via pre-commit-setup (commit-msg hook installed)
- [ ] protected-tag settings allow the pipeline to create `v*` tags
- [ ] `git cliff --bumped-version` sanity-checked locally (if git-cliff available)
- [ ] first default-branch pipeline green; releasable commit produces tag + release with notes
