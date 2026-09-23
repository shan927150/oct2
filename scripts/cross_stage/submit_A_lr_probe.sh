#!/bin/bash
# Explicit prepare/test/run/status. No automatic downstream submission.
# Run from the independent LR-probe clone, not the active E95/E100 worktree.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
exec python3 scripts/cross_stage/23_A_lr_control.py "$@"
