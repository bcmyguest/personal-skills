# ski — skill-inject

Local, model-agnostic **automatic skill injection** for Claude Code and opencode.
Embeds the prompt locally, ranks it against skill descriptions, and (later
milestones) injects the matched skill into context with no `use_skill` tool call.

See [PLAN.md](./PLAN.md) for the full design.

## Status — milestone 2

Skill discovery + embedding index + hybrid ranking (`ski index` / `ski why`) plus
the hook hot-path `ski hook`: reads a hook event on stdin, ranks, dedups against
per-session state, and writes the host's injection contract on stdout. `observe`
and `session-start` are stubbed for milestone 3; `init` lands later.

`ski hook` fails open by design — bad stdin, a missing index, or any IO error
yields an empty injection and exit 0, never a blocked prompt.

## Build & test

```sh
cargo build            # offline: bag-of-words embedder, no model download
cargo test             # unit + golden tests (real repo skills)
cargo run -- index     # build the index at $XDG_DATA_HOME/ski/index.json
cargo run -- why "how should I credit Claude in this commit" --top 5

# Hook hot-path (stdin event -> injection JSON on stdout):
echo '{"session_id":"s1","cwd":".","prompt":"credit Claude in this commit"}' \
  | cargo run -- hook --host claude     # -> {"hookSpecificOutput":{...,"additionalContext":...}}
echo '{"session_id":"s1","cwd":".","prompt":"set up a python project"}' \
  | cargo run -- hook --host opencode   # -> {"skills":[...],"inject":"..."}
```

Session state (the dedup ledger) lives at
`$XDG_STATE_HOME/ski/sessions/<session_id>.json`; a skill already injected — or
loaded by the model itself — is never re-injected within a session.

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
