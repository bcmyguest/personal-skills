---
name: gitlab-release
description: Add automated releases with generated release notes to a GitLab project — every push to the default branch computes the next semver from Conventional Commits via git-cliff, creates the vX.Y.Z tag, and publishes a GitLab Release with notes rendered from the commits. Ships the CI include file and cliff.toml to copy verbatim. Use when asked to automate releases, tagging, changelogs, or release notes on a GitLab repo. For GitHub repos the equivalent release workflows ship inside rust-setup and uv-setup.
---

# Automated GitLab releases from Conventional Commits

Both files ship complete in [`templates/`](templates/) and are
language-agnostic — no placeholders, copy verbatim:

```bash
cp "$SKILL/templates/cliff.toml" .                 # commit types → semver + notes
cp "$SKILL/templates/release.gitlab-ci.yml" .      # the two release jobs
```

## Wire it into `.gitlab-ci.yml`

Add the include, and make sure a `release` stage exists (or rename the jobs'
stage to one that does):

```yaml
stages: [..., release]

include:
  - local: release.gitlab-ci.yml
```

How it runs: on every push to the default branch, `release:plan` (git-cliff
image) computes the bumped version from the conventional commits since the
last tag and renders `release-notes.md`; `release:publish` (release-cli
image) then creates the tag and the GitLab Release with those notes. If the
bumped version equals the latest tag — nothing releasable landed — both jobs
no-op. The first releasable commit produces `v0.1.0` (cliff.toml's
`initial_tag`).

## Enforce Conventional Commits

The whole scheme runs on commit types, so enforce them at commit time: follow
the **pre-commit-setup** skill, including its `conventional-commits.repos.yaml`
fragment in the assembled config (and the `--hook-type commit-msg` install).

## Caveats

- **Protected tags:** if the project protects `v*` tags
  (Settings → Repository → Protected tags), the job token can only create
  them when the pipeline runner is allowed to — grant Maintainers "create"
  on the pattern or relax it.
- **Full history:** the plan job sets `GIT_DEPTH: 0`; don't override it with
  a shallow clone setting, git-cliff needs the tags.
- **Publishing packages** is out of scope here, but both crates.io and PyPI
  support **Trusted Publishing from GitLab CI** (OIDC id_tokens — never a
  long-lived registry token in CI variables); consult the registry's trusted
  publisher docs and append a publish job gated on `RELEASE == "true"`.

## Verify

- Locally, with git-cliff installed: `git cliff --bumped-version` prints the
  version the next default-branch push would release; `git cliff --unreleased
  --tag <that-version>` previews the notes.
- After pushing the wiring, CI → Editor → Validate (or the pipeline lint API)
  should accept the config, and the first pipeline on the default branch
  should show both jobs (no-op until a releasable commit lands).

## Checklist

- [ ] `cliff.toml` + `release.gitlab-ci.yml` copied verbatim to the repo root
- [ ] include added; `release` stage present in `stages:`
- [ ] Conventional Commits enforced via pre-commit-setup (commit-msg hook installed)
- [ ] protected-tag settings allow the pipeline to create `v*` tags
- [ ] `git cliff --bumped-version` sanity-checked locally (if git-cliff available)
- [ ] first default-branch pipeline green; releasable commit produces tag + release with notes
