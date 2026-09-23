#!/bin/bash
# Included in the delivery ZIP beside CODE, SHA256SUMS and the incremental bundle.
# Creates a fresh independent clone. Never updates the active E95/E100 checkout.
set -euo pipefail
SRC="${1:?usage: install_A_lr_probe_isolated.sh SOURCE_ROUTE_A NEW_CLONE}"
DEST="${2:?pass a new, nonexistent clone directory}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE=3231a195b32276f69ddc52a4a163eac1d9ee9b4d
BRANCH=exp/pathway2-a-stationarity
NEW="$(tr -d '\r\n' < "$HERE/CODE")"
[[ "$NEW" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid package CODE" >&2; exit 2; }
(
  cd "$HERE"
  sha256sum -c SHA256SUMS
)
test ! -e "$DEST" || { echo "Destination already exists; it will not be overwritten: $DEST" >&2; exit 2; }
python3 - "$SRC" "$DEST" <<'PY'
from pathlib import Path
import sys
src, dest = (Path(p).expanduser().resolve() for p in sys.argv[1:])
if src == dest or src in dest.parents or dest in src.parents:
    raise SystemExit("Source and destination must be separate directories")
PY
test "$(git -C "$SRC" rev-parse HEAD)" = "$BASE" || {
  echo "Active Route A source is not at $BASE. Stop and inspect it; nothing will be changed." >&2; exit 2;
}
test "$(git -C "$SRC" branch --show-current)" = "$BRANCH" || { echo "Wrong source branch" >&2; exit 2; }
BEFORE_STATUS="$(git -C "$SRC" status --porcelain --untracked-files=no)"
test -z "$BEFORE_STATUS" || { echo "Source has tracked edits; stop without changing them." >&2; exit 2; }
git clone --no-hardlinks --branch "$BRANCH" "$SRC" "$DEST"
git -C "$DEST" remote remove origin
git -C "$DEST" bundle verify "$HERE/oct2_A_lr_probe.bundle"
git -C "$DEST" fetch "$HERE/oct2_A_lr_probe.bundle" "$BRANCH"
test "$(git -C "$DEST" rev-parse FETCH_HEAD)" = "$NEW"
git -C "$DEST" merge --ff-only "$NEW"
test ! -s "$DEST/.git/objects/info/alternates"
test "$(git -C "$SRC" rev-parse HEAD)" = "$BASE"
test "$(git -C "$SRC" status --porcelain --untracked-files=no)" = "$BEFORE_STATUS"
test "$(git -C "$DEST" rev-parse HEAD)" = "$NEW"
printf 'Installed independent clone: %s\nCODE=%s\nOriginal Route A remains at %s\nNo jobs submitted.\n' "$DEST" "$NEW" "$BASE"
