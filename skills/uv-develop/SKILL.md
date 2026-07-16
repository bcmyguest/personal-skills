---
name: uv-develop
description: Fixes for uv dependency and lockfile operations an agent otherwise gets wrong — upgrading one package without churning the rest, resolving "lockfile out of date" / stale-lock CI failures, and repairing a broken or drifted .venv. Use when upgrading, bumping, or pinning a dependency; when uv reports the lockfile is stale or out of date; when uv sync/uv run fails on the lock; or when the venv is broken, drifted, or has untracked packages. Corrective notes only — routine uv add / uv remove / uv run need no skill.
---

# uv: the operations models get wrong

Routine `uv add <pkg>`, `uv remove <pkg>`, `uv run <cmd>` are correct as-is — do them directly.
This skill covers only the cases where the obvious command is the wrong one.

## Upgrade ONE package — not the whole world

To bump a single dependency, upgrade **that package only** and re-sync:

```bash
uv lock --upgrade-package <pkg>     # re-resolve just <pkg> (and only what it forces)
uv sync                             # apply to the venv
```

Do **not** reach for blanket `uv lock --upgrade` / `-U` to "bump requests" — it re-resolves
*every* dependency and produces a huge, unreviewable lock diff. Use the blanket form only when the
explicit intent is "update everything." Either way, review the `uv.lock` diff and run the tests
before committing.

`-P` / `--upgrade-package` accepts a constraint too: `uv lock --upgrade-package 'requests>=2.32'`.

## Lockfile freshness: `--locked` vs `--frozen` vs bare `uv sync`

These are three different behaviors — pick deliberately:

| You want | Command | What it does |
| --- | --- | --- |
| Reproduce exactly, **fail if lock is stale** (CI, clone verification) | `uv sync --locked` | Asserts `uv.lock` is up-to-date with `pyproject.toml`; **errors out** if not. |
| Install from the lock, **skip the freshness check** | `uv sync --frozen` | Uses the lock as source of truth without re-resolving or checking; won't catch a stale lock. |
| Normal local sync | `uv sync` | **Re-locks first** (may rewrite `uv.lock`), then installs. |

The trap: a bare `uv sync` re-locks before syncing, so it can silently rewrite `uv.lock`. In CI and
when verifying a checkout, use **`--locked`** — you want the run to *fail* on a stale lock, not
paper over it. Reach for `--frozen` only when you knowingly want to install the committed lock and
skip the check (e.g. an offline or fast path where you trust the lock).

`uv run` takes the same flags: `uv run --locked -m pytest` runs the suite and fails if the lock
drifted.

## "Lockfile out of date" / resolution errors

When uv complains the lock is out of date, the fix is to regenerate it through uv, then sync:

```bash
uv lock        # re-resolve pyproject.toml -> uv.lock
uv sync        # bring the venv in line
```

Never hand-edit `uv.lock` to silence the error, and never hand-edit `pyproject.toml` dependency
tables and then expect the lock to follow — add/remove/upgrade through uv so both files stay
consistent.

## Broken, drifted, or "why is my venv weird" environments

- **Don't `uv pip install <pkg>` inside a uv project.** It mutates `.venv` without touching
  `pyproject.toml` or `uv.lock`, so the change is untracked and vanishes on the next `uv sync`.
  Use `uv add <pkg>` (or `uv add --dev <pkg>`) instead.
- **Untracked packages / drift:** re-sync from the lock. `uv sync` removes anything not in the lock
  by default, restoring the environment to exactly what's committed.
- **Genuinely corrupt venv:** delete and rebuild — `rm -rf .venv && uv sync`. uv recreates it from
  `uv.lock`.
- **Wrong interpreter:** the venv hard-codes its Python path. If the pinned version changed
  (`.python-version`), `rm -rf .venv && uv sync` rebuilds against the correct interpreter.

## Quick reference (corrective cases only)

| Situation | Command |
| --- | --- |
| Bump one dependency | `uv lock --upgrade-package <pkg> && uv sync` |
| Bump everything (deliberate) | `uv lock --upgrade && uv sync` |
| CI / verify a clone (fail on stale lock) | `uv sync --locked` |
| Regenerate a stale lock | `uv lock && uv sync` |
| Restore env to the committed lock | `uv sync` (or `rm -rf .venv && uv sync`) |
| Added a package by accident via pip | undo, then `uv add <pkg>` |
