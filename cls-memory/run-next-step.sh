#!/usr/bin/env bash
#
# Unattended P0 run: the two measurements that decide the recall work.
#
#     ./run-next-step.sh
#
# Sets up the venv, fetches both datasets if missing, then runs:
#   1. the full recall ablation, including the random-projection rows that
#      test whether a JL projection beats truncated SVD (it does, on a
#      one-conversation smoke test: RP-4096 scored 0.327@1 against LSA-1024's
#      0.250 and sparse TF-IDF's 0.311)
#   2. the cumulative before/after check on BOTH corpora, with a widened
#      QMSum slice so the verdict is not resting on 38 questions
#
# Idempotent and safe to re-run. Everything lands in results/ with a timestamp;
# a summary is printed at the end. Expect 20-40 minutes.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="results/${STAMP}"
mkdir -p "$OUT"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- environment
if [ ! -x "$PYTHON" ]; then
  log "creating venv"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
    uv pip install --python "$PYTHON" torch pytest
  else
    python3 -m venv .venv
    "$PYTHON" -m pip install --quiet --upgrade pip
    "$PYTHON" -m pip install --quiet torch pytest
  fi
else
  log "using existing venv"
fi

# ------------------------------------------------------------------- datasets
mkdir -p experiments/data
fetch() {  # fetch <path> <url>
  if [ -s "$1" ]; then
    echo "    have $(basename "$1")"
  else
    echo "    fetching $(basename "$1")"
    curl -fsSL --retry 3 -o "$1" "$2"
  fi
}
log "datasets"
fetch experiments/data/locomo10.json \
  "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
fetch experiments/data/qmsum_test.jsonl \
  "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/test.jsonl"

# ---------------------------------------------------------------------- tests
log "test suite (sanity — nothing below is meaningful if this fails)"
if PYTHONPATH=. "$PYTHON" -m pytest tests/ -q > "$OUT/tests.log" 2>&1; then
  tail -1 "$OUT/tests.log"
else
  tail -20 "$OUT/tests.log"
  echo "TESTS FAILED — stopping. See $OUT/tests.log"
  exit 1
fi

# ------------------------------------------------------------------ P0.2 first
# Run the ablation before the check: it is the decision-making measurement, so
# if the run is interrupted this is the part you want to have finished.
log "P0.2  recall ablation — where the signal goes, and can RP beat the SVD"
PYTHONPATH=. "$PYTHON" -u experiments/recall_ablation.py --conversations 3 \
  > "$OUT/ablation.log" 2>&1 || true
grep -v "NumPy\|cpu = _conv\|UserWarning\|^  return torch\|^  lsa = " "$OUT/ablation.log" || true

# ---------------------------------------------------------------- P0.1 + check
log "P0.1/3.5  cumulative recall check on BOTH corpora, widened QMSum"
PYTHONPATH=. "$PYTHON" -u experiments/recall_check.py --locomo 3 --qmsum 25 \
  > "$OUT/recall_check.log" 2>&1 || true
grep -v "NumPy\|cpu = _conv\|UserWarning\|^  embedder = " "$OUT/recall_check.log" || true

# -------------------------------------------------------------------- summary
log "summary"
cat <<SUMMARY
Logs: $OUT/

How to read this:

  ablation.log, "RETRIEVAL CEILINGS" block
    Compare random-projection-* against LSA-1024 and against TF-IDF cosine.
    - RP clearly above LSA-1024   -> promote it. See HANDOFF.md 3A task 1.1;
                                     the class is experiments/recall_ablation.py
                                     HashedProjection, which already satisfies
                                     the Embedder protocol. Move it into
                                     cls_memory/embeddings.py, make it the
                                     default, add tests, re-run this script.
    - RP at or below LSA-1024    -> abandon 1.1, go to the hybrid shortlist
                                     (HANDOFF.md 3A task 1.2).

  ablation.log, "FULL PIPELINE" block
    Any pipeline row should sit close to its embedder's ceiling. A large gap
    means the memory system started losing signal again — that would be new.

  recall_check.log
    The 'now:' row is the current default. Beating 0.320@1 on LoCoMo means you
    have passed the sparse-lexical ceiling. Check BOTH corpora move the same
    way; a change that helps one and hurts the other is overfitting, and that
    has already happened once here.

Then read HANDOFF.md 3A for what follows.
SUMMARY
