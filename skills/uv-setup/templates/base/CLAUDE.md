# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

{{project_name}}: {{description}}

<!-- Rewrite the description and add an Architecture section once real code
     replaces the hello-world mock (`/init` can draft it). Keep the Commands
     and Releases sections — they encode the repo's conventions. -->

## Commands

Everything runs through uv — never bare `python`, `pip`, `virtualenv`, or
`pytest`:

```bash
uv sync --frozen                # reproduce the locked environment
uv run -m pytest                # tests (after every change)
uv run ruff check               # lint
uv run ruff format --check      # formatting (CI enforces it)
uv run pyrefly check            # types
uv add <pkg> / uv add --dev <pkg>   # change deps — never hand-edit uv.lock
```

Pre-commit enforces Conventional Commits (`feat:`, `fix:`, `docs:`, …) via a
commit-msg hook; install with
`uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg`.
Commit types drive release versioning, so pick them deliberately.

<!-- DELETE this section if you skipped release automation; drop the PyPI
     sentences if release-github.yml (no publishing) was chosen. -->
## Releases

Every merge to `main` runs `.github/workflows/release.yml`: git-cliff computes
the next semver from conventional commits (`feat` → minor, breaking → major,
else patch), pushes the `vX.Y.Z` tag, publishes a GitHub release, and builds
sdist + wheel and uploads to PyPI via Trusted Publishing (OIDC; the PyPI
project must list this repo + `release.yml` + environment `pypi` as a trusted
publisher or the publish job fails the run — the tag and GitHub release land
first regardless). Running the workflow manually (`workflow_dispatch`) skips
the release half and (re)publishes the latest existing tag to PyPI, e.g. to
backfill after a failed publish. The committed `pyproject.toml` version is
never bumped — the workflow stamps the tag version in at build time. Don't
bump the version in PRs.
