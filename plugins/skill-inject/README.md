# ski — skill-inject

Local, model-agnostic **automatic skill injection** for Claude Code and opencode.
Embeds the prompt locally, ranks it against skill descriptions, and (later
milestones) injects the matched skill into context with no `use_skill` tool call.

See [PLAN.md](./PLAN.md) for the full design.

## Status — milestone 1

Skill discovery + embedding index + hybrid ranking, exposed as `ski index` and
`ski why`. The hook hot-path (`hook` / `observe` / `session-start`) and `init` are
stubbed for milestone 2.

## Build & test

```sh
cargo build            # offline: bag-of-words embedder, no model download
cargo test             # unit + golden tests (real repo skills)
cargo run -- index     # build the index at $XDG_DATA_HOME/ski/index.json
cargo run -- why "how should I credit Claude in this commit" --top 5
```

## Embedding backends

- **Default (offline):** deterministic hashed bag-of-words. No deps, no network,
  no model. Surface-token matching only; the keyword boost covers exact terms.
  Used for tests and as a fallback.
- **`--features fastembed`:** real embeddings via fastembed (ONNX). Default model
  `bge-small-en-v1.5` (query gets bge's retrieval-instruction prefix; descriptions
  don't), lite alternative `all-MiniLM-L6-v2-q`. Model downloads once and is cached.

  ```sh
  cargo build --release --features fastembed
  ```

The index is tagged with the embedder id, so switching backends/models triggers a
full reindex automatically.

## Lint

`cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings`. Also
wired as `pre-commit` hooks at the repo root (`ski-fmt` / `ski-clippy` / `ski-test`).
