# python-dev

Python project tooling built around [uv](https://docs.astral.sh/uv/).

| Skill | What it does |
|-------|--------------|
| `uv-setup` | Bootstrap a new uv project — packaged layout, pinned Python, type checker (pyright), ruff + pyright pre-commit hooks, run/build/validate docs, optional Dockerfile. |
| `uv-develop` | Day-to-day uv workflow — `uv sync --frozen`, `uv run -m pytest`, `uv add/remove`. Always prefer uv over pip/poetry/pipenv. |
| `pre-commit-setup` | Stand up pre-commit with the standard hygiene hooks and install the git hook. Project linters are layered on by `uv-setup`. |

Skills live in [`../skills/`](../skills); this plugin selects them via `skills` in the
[marketplace manifest](../.claude-plugin/marketplace.json).
