#!/bin/bash
# Run from the reviewed worktree on Delta. First argument: stable (default) or legacy background.
set -euo pipefail
pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
recipe="${1:-stable}"
case "$recipe" in stable|legacy) ;; *) echo 'Usage: bash start_delta.sh [stable|legacy]' >&2; exit 2 ;; esac
module reset
module load pytorch-conda/2.8
export PYTHONDONTWRITEBYTECODE=1
runs=/u/yli103/oct2-pathway2-runs/A
out="$runs/cross_stage_lds_v4_${recipe}_$(date -u +%Y%m%dT%H%M%S%NZ)"
python3 "$pkg/run.py" prepare --out "$out" --background-recipe "$recipe"
printf '%s\n' "$out" > "$runs/cross_stage_lds_v4_${recipe}_current.txt"
python3 "$pkg/run.py" submit --out "$out"
printf 'OUT=%s\n' "$out"
