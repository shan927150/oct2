#!/bin/bash
#SBATCH --job-name=oct_e0b0_test
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:20:00
#SBATCH --output=logs/oct_e0b0_test-%j.out
#SBATCH --error=logs/oct_e0b0_test-%j.err
set -euo pipefail
cd "${OCT_B_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/cross_stage/pathway2_job_env.sh
for TEST in test_e0_residual_decomposition test_b0_dose_derivative test_b0_train test_stage1_v11; do
  python3 -B "scripts/cross_stage/tests/$TEST.py"
done
echo "E0_B0_GPU_TESTS_PASSED"
