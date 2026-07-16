# Multi-stage build: the final image gets the app and its venv but NOT uv.
# Based on https://github.com/astral-sh/uv-docker-example (multistage.Dockerfile).
#
# THE TRAP — Python versions in lockstep. The builder image, the final image,
# and .python-version must all be the SAME Python release. The builder copies
# /app/.venv into the final image, and the venv hard-codes its interpreter
# path; if the final image's Python differs (even 3.12 vs 3.13) the app fails
# to start. If you bump the pinned Python later, update BOTH FROM lines and
# re-test.
FROM ghcr.io/astral-sh/uv:python{{python_version}}-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_NO_DEV=1               # omit dev dependencies
ENV UV_PYTHON_DOWNLOADS=0     # use the image's interpreter, don't fetch one

WORKDIR /app
# Install deps first (cached layer) without the project, then the project
# itself. `uv sync --locked` is the container equivalent of local `--frozen`:
# install from uv.lock, fail if the lock is stale — the image always matches
# what's committed.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Final stage: plain python image — MUST match the builder's Python version
# (see the trap above).
FROM python:{{python_version}}-slim-trixie

RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

COPY --from=builder --chown=nonroot:nonroot /app /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
USER nonroot
WORKDIR /app

# The console script from [project.scripts]; swap for the real server command
# when one exists (e.g. ["uvicorn", "{{package_snake}}.api:app", "--host", "0.0.0.0"]).
CMD ["{{project_name}}"]
