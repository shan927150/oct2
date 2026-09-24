#!/bin/bash
#SBATCH --job-name=oct_A_sp_test
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_DIR:?}"
module reset
module load pytorch-conda/2.8
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 STAGE1_DIAG_REQUIRE_CUDA=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD/src:$PWD/scripts/cross_stage${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/cross_stage/24_A_smallpanel_control.py check "${OCT_A_SP_OUT:?}"
srun python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.__version__, torch.cuda.get_device_name())'
srun python3 -m unittest discover -s scripts/cross_stage/tests -p 'test_A_smallpanel*.py' -v
srun python3 scripts/cross_stage/tests/test_A_endpoint_mc.py
srun python3 scripts/cross_stage/24_A_smallpanel_control.py smoke "$OCT_A_SP_OUT"
python3 scripts/cross_stage/24_A_smallpanel_control.py mark-test "$OCT_A_SP_OUT"
echo A_SMALLPANEL_GPU_AND_REAL_E100_TESTS_PASSED
