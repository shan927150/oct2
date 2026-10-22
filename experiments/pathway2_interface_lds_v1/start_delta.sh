#!/bin/bash
# Run from the committed worktree on Delta: prepare one new OUT from the completed v6 r1 run, then submit.
# Usage: bash start_delta.sh [V6_OUT] [extra prepare options, e.g. --budget-gpu-hours 24]
set -euo pipefail
pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
module reset
module load pytorch-conda/2.8
export PYTHONDONTWRITEBYTECODE=1
runs=/u/yli103/oct2-pathway2-runs/A
v6="${1:-$(cat "$runs/cross_stage_lds_v6_current.txt")}"
shift || true
out="$runs/interface_lds_v1_$(date -u +%Y%m%dT%H%M%S%NZ)"
python3 "$pkg/run.py" prepare --out "$out" --v6-out "$v6" "$@"
printf '%s\n' "$out" > "$runs/interface_lds_v1_current.txt"
python3 "$pkg/run.py" submit --out "$out"
printf 'OUT=%s\n' "$out"
