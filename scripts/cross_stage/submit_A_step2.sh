#!/bin/bash
# Manually gated Route-A Step 2. Every submit command creates exactly one
# Slurm job and never adds an automatic dependency. Review `status` and the
# completed artifacts before invoking the next command.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit_A_step2.sh prepare SELECTION_JSON
  submit_A_step2.sh test [STEP2_ROOT]
  submit_A_step2.sh truth-full [STEP2_ROOT]
  submit_A_step2.sh score [STEP2_ROOT]
  submit_A_step2.sh compare-primary [STEP2_ROOT]
  submit_A_step2.sh truth-dose [STEP2_ROOT]
  submit_A_step2.sh preflight [STEP2_ROOT]
  submit_A_step2.sh approve-a2 [STEP2_ROOT] NOTE
  submit_A_step2.sh damping [STEP2_ROOT]
  submit_A_step2.sh compare-full [STEP2_ROOT]
  submit_A_step2.sh status [STEP2_ROOT]

prepare requires OCT_BASELINE_ROOT and OCT_RUNS_ROOT. GPU stages additionally
require OCT_DATA_DIR. STEP2_ROOT may instead be exported as OCT_A_STEP2_ROOT.
Nothing downstream is submitted automatically.
EOF
}

COMMAND="${1:-}"
if [ -z "$COMMAND" ] || [ "$COMMAND" = "-h" ] || [ "$COMMAND" = "--help" ]; then
  usage
  exit 0
fi
shift

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
CONTROL="scripts/cross_stage/21_A_step2_control.py"
ACCOUNT="${OCT_ACCOUNT:-bgjy-delta-gpu}"

root_arg() {
  local supplied="${1:-${OCT_A_STEP2_ROOT:-}}"
  if [ -z "$supplied" ]; then
    echo "Pass STEP2_ROOT or export OCT_A_STEP2_ROOT" >&2
    exit 2
  fi
  python3 - "$supplied" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

load_state() {
  export OCT_A_STEP2_ROOT="$1"
  export OCT_A_E_STAR
  OCT_A_E_STAR="$(python3 "$CONTROL" field --root "$1" --name E_star)"
  export OCT_A_CODE
  OCT_A_CODE="$(python3 "$CONTROL" field --root "$1" --name git_commit)"
  export OCT_BASELINE_ROOT
  OCT_BASELINE_ROOT="$(python3 "$CONTROL" field --root "$1" --name baseline_root)"
  export OCT_A_SELECTION="$1/frozen_E_star.json"
  export PROJECT_DIR="$REPO"
}

submit_one() {
  local action="$1"
  shift
  python3 "$CONTROL" check --root "$OCT_A_STEP2_ROOT" --action "$action" >/dev/null
  local job
  job="$(sbatch --parsable --account="$ACCOUNT" "$@")"
  job="${job%%;*}"
  python3 "$CONTROL" record-job --root "$OCT_A_STEP2_ROOT" --action "$action" --job_id "$job"
  echo "$action submitted as job $job"
  echo "No downstream job was submitted. Inspect with:"
  echo "  bash scripts/cross_stage/submit_A_step2.sh status '$OCT_A_STEP2_ROOT'"
}

case "$COMMAND" in
  prepare)
    SELECTION="${1:?prepare requires frozen_E_star.json}"
    : "${OCT_BASELINE_ROOT:?set OCT_BASELINE_ROOT}"
    : "${OCT_RUNS_ROOT:?set OCT_RUNS_ROOT (must end in /A)}"
    ROOT="$(python3 "$CONTROL" prepare --selection "$SELECTION" \
      --baseline_root "$OCT_BASELINE_ROOT" --runs_root "$OCT_RUNS_ROOT")"
    echo "Prepared immutable Step-2 attempt: $ROOT"
    printf "export OCT_A_STEP2_ROOT='%s'\n" "$ROOT"
    echo "Next, submit only the Step-2 GPU replay smoke test."
    ;;
  status)
    ROOT="$(root_arg "${1:-}")"
    python3 "$CONTROL" status --root "$ROOT"
    EVENTS="$ROOT/submission_events.jsonl"
    if [ -s "$EVENTS" ] && command -v sacct >/dev/null 2>&1; then
      IDS="$(python3 - "$EVENTS" <<'PY'
import json, sys
print(",".join(row["job_id"] for row in map(json.loads, open(sys.argv[1]))))
PY
)"
      if [ -n "$IDS" ]; then
        sacct -X -j "$IDS" --format=JobID%18,JobName%18,State,ExitCode,Elapsed,NodeList || true
      fi
    fi
    ;;
  test)
    ROOT="$(root_arg "${1:-}")"
    load_state "$ROOT"
    mkdir -p logs
    submit_one test scripts/cross_stage/21_A_tests.slurm
    ;;
  truth-full|truth-dose|score|preflight|damping)
    ROOT="$(root_arg "${1:-}")"
    load_state "$ROOT"
    : "${OCT_DATA_DIR:?set OCT_DATA_DIR}"
    mkdir -p logs
    case "$COMMAND" in
      truth-full) submit_one truth-full scripts/cross_stage/21_A_truth_at_E.slurm full ;;
      truth-dose) submit_one truth-dose scripts/cross_stage/21_A_truth_at_E.slurm dose01 ;;
      score) submit_one score scripts/cross_stage/21_A_score_at_E.slurm ;;
      preflight) submit_one preflight scripts/cross_stage/21_A_preflight_at_E.slurm ;;
      damping) submit_one damping scripts/cross_stage/21_A_damping_at_E.slurm ;;
    esac
    ;;
  compare-primary|compare-full)
    ROOT="$(root_arg "${1:-}")"
    load_state "$ROOT"
    ACTION="$COMMAND"
    MODE=primary
    OUT="$ROOT/compare_primary_vs_E50"
    if [ "$COMMAND" = "compare-full" ]; then
      MODE=full
      OUT="$ROOT/compare_full_vs_E50"
    fi
    python3 "$CONTROL" check --root "$ROOT" --action "$ACTION" >/dev/null
    module reset
    module load pytorch-conda/2.8
    python3 scripts/cross_stage/22_A_compare_E.py --mode "$MODE" \
      --new_root "$ROOT" --baseline_root "$OCT_BASELINE_ROOT" --out "$OUT"
    echo "Comparison written to $OUT"
    ;;
  approve-a2)
    if [ "$#" -ge 2 ]; then
      ROOT="$(root_arg "$1")"
      shift
      NOTE="$*"
    elif [ "$#" -eq 1 ] && [ -n "${OCT_A_STEP2_ROOT:-}" ]; then
      ROOT="$(root_arg "$OCT_A_STEP2_ROOT")"
      NOTE="$1"
    else
      echo "approve-a2 requires STEP2_ROOT and a quoted review note (or exported OCT_A_STEP2_ROOT)" >&2
      exit 2
    fi
    load_state "$ROOT"
    python3 "$CONTROL" approve-a2 --root "$ROOT" --note "$NOTE"
    echo "A2 is now unlocked for this attempt only. No damping job was submitted."
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage >&2
    exit 2
    ;;
esac
