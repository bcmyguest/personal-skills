#!/usr/bin/env bash
# Install the Claude Code and opencode plugins declared in plugins.json.
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
  # Try Claude
  if claude plugin marketplace list 2>/dev/null | grep -q "$name"; then
    claude plugin marketplace update "$name"
  else
    claude plugin marketplace add "$src" 2>/dev/null || true
  fi
  # Try opencode
  if opencode plugin marketplace list 2>/dev/null | grep -q "$name"; then
    opencode plugin marketplace update "$name"
  else
    opencode plugin marketplace add "$src" 2>/dev/null || true
  fi
done

# Install new, bump already-installed to the latest committed commit, ensure enabled.
python3 -c "import json;[print(p) for p in json.load(open('$cfg'))['enable']]" | while read -r p; do
  echo "==> $p"
  # Try Claude
  claude plugin install "$p" 2>/dev/null || true
  claude plugin update  "$p" 2>/dev/null || true
  claude plugin enable  "$p" 2>/dev/null || true

  # Try opencode
  opencode plugin install "$p" 2>/dev/null || true
  opencode plugin update  "$p" 2>/dev/null || true
  opencode plugin enable  "$p" 2>/dev/null || true
done

echo "--- installed plugins ---"
claude plugin list 2>/dev/null | grep "❯" || true
opencode plugin list 2>/dev/null | grep "❯" || true
echo "done. Restart your tools to apply plugin updates."
