#!/bin/bash
#SBATCH --job-name=oct_b1_test
#SBATCH --account=bgjy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=logs/oct_b1_test-%j.out
#SBATCH --error=logs/oct_b1_test-%j.err
# The regression suite on the GPU that will run the job. CPU-passing is not enough:
# the E0 hotfix history is exactly a case where a CUDA kernel, not the algebra, set the
# floor. Run it before any B1 array task is allowed to start.
set -euo pipefail
cd "${OCT_B_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/cross_stage/pathway2_job_env.sh
for TEST in test_b1_smoke test_e0_residual_decomposition test_b0_dose_derivative \
            test_b0_train test_stage1_v11; do
  python3 -B "scripts/cross_stage/tests/$TEST.py"
done
echo "B1_GPU_TESTS_PASSED"
