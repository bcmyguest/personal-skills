# {{project_name}}

[![CI](https://github.com/{{repo_slug}}/actions/workflows/ci.yml/badge.svg)](https://github.com/{{repo_slug}}/actions/workflows/ci.yml)
<!-- DELETE the PyPI badge if not publishing to PyPI -->
[![PyPI](https://img.shields.io/pypi/v/{{project_name}}.svg)](https://pypi.org/project/{{project_name}}/)

{{description}}

## Install

<!-- KEEP the section(s) matching your decisions; delete the rest. -->

<!-- IF library on PyPI -->
```bash
uv add {{project_name}}            # or: pip install {{project_name}}
```

<!-- IF application/CLI on PyPI -->
```bash
uv tool install {{project_name}}   # or run without installing: uvx {{project_name}}
```

<!-- IF service (container) -->
```bash
docker build -t {{project_name}} .
docker run --rm {{project_name}}
```

## Usage

<!-- Replace with real usage once the hello-world mock is gone. -->

```bash
{{project_name}} Ferris
# Hello, Ferris!
```

## Development

Everything runs through uv:

```bash
uv sync --frozen                # reproduce the locked environment
uv run -m pytest                # tests
uv run ruff check && uv run ruff format --check
uv run pyrefly check            # types
uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org)
(`feat:`, `fix:`, `docs:`, …) — enforced by a commit-msg hook, and the commit
type drives release versioning.

<!-- DELETE this section if nothing is distributed (service / internal app) -->
## Build & validate

```bash
uv build                        # sdist + wheel into dist/

# Prove the wheel installs and imports in a clean env (no project, no venv):
uv run --isolated --no-project --with dist/{{package_snake}}-*.whl \
  python -c "from {{package_snake}} import greet; print(greet('wheel'))"
```

<!-- DELETE this section if you skipped release automation -->
## Releases

Every merge to `main` runs `release.yml`: git-cliff computes the next semver
from the conventional commits since the last tag (`feat` → minor, breaking →
major, else patch), pushes the `vX.Y.Z` tag, and publishes a GitHub release
with generated notes.
<!-- DELETE the next sentence if not publishing to PyPI -->
The same run builds sdist + wheel and uploads them to PyPI via Trusted
Publishing (OIDC — no tokens anywhere).
The committed `pyproject.toml` version is never bumped by hand — the workflow
stamps the tag version in at build time.

## License

{{license}}
