#!/usr/bin/env bash
# Install the Claude Code plugins declared in plugins.json.
# Idempotent: safe to re-run; run it after cloning this repo on a new machine.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cfg="$repo/plugins.json"

# Add or update each marketplace. type=local -> source is the repo root (where
# .claude-plugin/marketplace.json lives); type=github -> source is "owner/repo".
python3 -c "
import json
for m in json.load(open('$cfg'))['marketplaces']:
    src = '$repo' if m['type']=='local' and m['path']=='.' else m['path']
    print(m['name'], src)
" | while read -r name src; do
  if claude plugin marketplace list 2>/dev/null | grep -q "$name"; then
    claude plugin marketplace update "$name"
  else
    claude plugin marketplace add "$src"
  fi
done

# Install new, bump already-installed to the latest committed commit, ensure enabled.
# `update` is the step that moves a local (directory) marketplace's cached plugin to
# the newest commit — install/enable alone won't, so already-installed plugins would
# stay pinned to whatever commit they were first installed at. All three are
# idempotent; failures (e.g. "already installed", "already enabled") are ignored.
python3 -c "import json;[print(p) for p in json.load(open('$cfg'))['enable']]" | while read -r p; do
  echo "==> $p"
  claude plugin install "$p" 2>/dev/null || true   # no-op if already installed
  claude plugin update  "$p" 2>/dev/null || true   # bump to latest commit (restart to apply)
  claude plugin enable  "$p" 2>/dev/null || true   # ensure enabled
done

echo "--- installed plugins ---"
claude plugin list 2>/dev/null | grep "❯" || true
echo "done. Restart Claude Code to apply plugin updates."
