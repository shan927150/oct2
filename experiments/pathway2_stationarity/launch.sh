#!/bin/bash
# Explicitly prepares and submits a new pilot or matrix; installation submits nothing.
set -euo pipefail
pkg="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:-pilot}"
case "$profile" in pilot|matrix) ;; *) printf 'Usage: bash launch.sh [pilot|matrix]\n' >&2; exit 2;; esac
export PYTHONDONTWRITEBYTECODE=1
root="${OCT_STATIONARITY_RUNS:-/u/yli103/oct2-pathway2-runs/A}"
out="$root/objective_stationarity_${profile}_$(date -u +%Y%m%dT%H%M%S%NZ)"
python3 "$pkg/run.py" prepare --out "$out" --profile "$profile" --reps 128
printf 'Save this output path: %s\n' "$out"
python3 "$pkg/run.py" submit --out "$out"
