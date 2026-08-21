#!/bin/bash
#SBATCH --job-name=oct_cs_loo05
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgjy-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/oct2}"
RUN_MODE="${RUN_MODE:-smoke}"
DATA_DIR="${DATA_DIR:-./data}"
cd "$PROJECT_DIR"
mkdir -p logs results

module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1

# Support both the GitHub layout (src/*.py) and Delta's historical flat
# deployment (config.py/data.py/models.py/split.py under ~/oct2).
if [ -f "$PROJECT_DIR/src/config.py" ]; then
  export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
elif [ -f "$PROJECT_DIR/config.py" ]; then
  export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
else
  echo "ERROR: cannot find config.py under $PROJECT_DIR/src or $PROJECT_DIR"
  exit 2
fi

echo "=== Cross-stage patient LOO 05 ==="
echo "job=$SLURM_JOB_ID node=$(hostname) mode=$RUN_MODE date=$(date)"
echo "project=$PROJECT_DIR data=$DATA_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY

SCRIPT=scripts/cross_stage/05_end_to_end_patient_loo_pilot.py
if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: $PROJECT_DIR/$SCRIPT not found"
  exit 2
fi

if [ "$RUN_MODE" = "smoke" ]; then
  python "$SCRIPT" \
    --output_dir ./results/cross_stage_patient_loo_05_smoke_v2 \
    --data_dir "$DATA_DIR" \
    --n_total_samples 5000 \
    --target_data_size 400 \
    --shadow_data_size 400 \
    --n_shadow 2 \
    --affected_shadow 0 \
    --n_patients 2 \
    --classes 1 2 \
    --min_patient_images 1 \
    --max_patient_images 30 \
    --seeds 42 \
    --shadow_epochs 5 \
    --attack_epochs 5 \
    --noop_replays 1
elif [ "$RUN_MODE" = "gate" ] || [ "$RUN_MODE" = "pilot" ]; then
  EXTRA_ARGS=()
  if [ "$RUN_MODE" = "gate" ]; then
    EXTRA_ARGS+=(--baseline_only)
  fi
  python "$SCRIPT" \
    --output_dir ./results/cross_stage_patient_loo_05_pilot_v2 \
    --data_dir "$DATA_DIR" \
    --n_total_samples 12000 \
    --target_data_size 800 \
    --shadow_data_size 800 \
    --n_shadow 5 \
    --affected_shadow 0 \
    --n_patients 8 \
    --classes 1 2 \
    --min_patient_images 5 \
    --max_patient_images 15 \
    --seeds 42 43 44 \
    --shadow_epochs 50 \
    --attack_epochs 50 \
    --noop_replays 1 \
    --noop_tolerance 0 \
    --enforce_attack_gate \
    --enforce_noop_gate \
    "${EXTRA_ARGS[@]}"
else
  echo "ERROR: RUN_MODE must be smoke, gate, or pilot, got '$RUN_MODE'"
  exit 2
fi

echo "=== Finished: $(date) ==="
