#!/bin/bash
# Run from the committed v6 worktree on Delta: prepare one new OUT on top of the v5 r1 run, then submit.
set -euo pipefail
pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
module reset
module load pytorch-conda/2.8
export PYTHONDONTWRITEBYTECODE=1
runs=/u/yli103/oct2-pathway2-runs/A
out="$runs/cross_stage_lds_v6_$(date -u +%Y%m%dT%H%M%S%NZ)"
python3 "$pkg/run.py" prepare --out "$out" "$@"
printf '%s\n' "$out" > "$runs/cross_stage_lds_v6_current.txt"
python3 "$pkg/run.py" submit --out "$out"
printf 'OUT=%s\n' "$out"
