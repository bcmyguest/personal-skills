#!/usr/bin/env bash
# Copy this repo's wteam/ folder (teams, phases, persona roles) into the wteam config dir.
# Overwrites files with the same name; re-run after `git pull`. Leaves other config alone.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/wteam"
DEST="${WTEAM_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/wteam}"

cd "$SRC"
find . -type f | sort | while read -r rel; do
  rel="${rel#./}"
  if cmp -s "$rel" "$DEST/$rel"; then
    status=unchanged
  elif [[ -e "$DEST/$rel" ]]; then
    status=updated
  else
    status=new
  fi
  mkdir -p "$DEST/$(dirname "$rel")"
  cp "$rel" "$DEST/$rel"
  echo "$status  $DEST/$rel"
done

if command -v wteam >/dev/null; then
  wteam teams
  wteam phases
fi
