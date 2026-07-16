# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

{{crate_name}}: {{description}}

<!-- Rewrite the description and add an Architecture section once real code
     replaces the hello-world mock (`/init` can draft it). Keep the Commands
     and Releases sections — they encode the repo's conventions. -->

## Commands

```bash
cargo test                                  # full suite: unit + integration
cargo clippy --all-targets -- -D warnings   # CI fails on any warning
cargo fmt --check
cargo build --release
```

Pre-commit enforces Conventional Commits (`feat:`, `fix:`, `docs:`, …) via a
commit-msg hook; install with
`pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg`.
Commit types drive release versioning, so pick them deliberately.

<!-- DELETE this section if you skipped release automation; trim the
     crates.io / binary sentences to match the release.yml variant chosen. -->
## Releases

Every merge to `main` runs `.github/workflows/release.yml`: git-cliff computes
the next semver from conventional commits (`feat` → minor, breaking → major,
else patch), pushes the `vX.Y.Z` tag, builds a static binary per supported
target, publishes a GitHub release, and publishes to crates.io via Trusted
Publishing (OIDC; the crate's crates.io settings must list this repo +
`release.yml` as a trusted publisher or the publish step fails the run — the
tag and GitHub release land first regardless). Running the workflow manually
(`workflow_dispatch`) skips the release half and (re)publishes the latest
existing tag to crates.io, e.g. to backfill after a failed publish. The
committed `Cargo.toml` version is never bumped — the workflow stamps the tag
version in at build time. Don't bump the version in PRs.
