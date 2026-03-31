#!/usr/bin/env bash
# ==============================================================================
# Slurm batch script for running ris_beam_selection_v8_3gpp.py on an A100 GPU.
#
# Usage:
#   sbatch run_ris_gpu.sbatch
#
# Submit from the repository root so relative paths resolve correctly.
# ==============================================================================

#SBATCH --job-name=ris_beam_selection
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=ris_job_%j.out
#SBATCH --error=ris_job_%j.err
# Uncomment and set your email address to receive job notifications:
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your_email@example.com

# ---------------------------------------------------------------------------
# Shell safety: exit on error, unbound variable, or pipeline failure.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable names (override on the sbatch command line if needed).
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-ris_env}"
PYTHON_SCRIPT="Claude_Promo_Period/ris_beam_selection_v8_3gpp.py"

# ---------------------------------------------------------------------------
# Per-job output directory.
# ---------------------------------------------------------------------------
OUTPUT_DIR="output_${SLURM_JOB_ID}"

# ---------------------------------------------------------------------------
# GPU stats logger PID (initialised to empty so the trap is always safe).
# ---------------------------------------------------------------------------
GPU_LOGGER_PID=""

# ---------------------------------------------------------------------------
# Cleanup trap — always runs on exit or signal, even on error.
# Stops the background nvidia-smi logger and records the final exit code.
# ---------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [[ -n "${GPU_LOGGER_PID}" ]] && kill -0 "${GPU_LOGGER_PID}" 2>/dev/null; then
        kill "${GPU_LOGGER_PID}"
    fi
    echo "=== Job finished at $(date) | exit code: ${exit_code} ===" \
        | tee -a "${OUTPUT_DIR}/run.log" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Runtime metadata.
# ---------------------------------------------------------------------------
echo "=== Job started at $(date) ==="
echo "    Job ID   : ${SLURM_JOB_ID}"
echo "    Host     : $(hostname)"
echo "    GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo "    Conda env: ${CONDA_ENV}"
echo "    Output   : ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Verify the Python script exists before doing anything else.
# ---------------------------------------------------------------------------
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    echo "       Run sbatch from the repository root." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Load Anaconda and activate the conda environment.
# ---------------------------------------------------------------------------
module load anaconda3
# shellcheck disable=SC1091  # conda.sh is dynamically located; path not known at lint time
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# Create per-job output directory and copy job artefacts into it.
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
cp "${PYTHON_SCRIPT}" "${OUTPUT_DIR}/"
cp "${BASH_SOURCE[0]}"  "${OUTPUT_DIR}/run_ris_gpu.sbatch"

# ---------------------------------------------------------------------------
# Start periodic GPU stats logger (every 30 s) in the background.
# ---------------------------------------------------------------------------
nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv \
    --loop=30 \
    >> "${OUTPUT_DIR}/gpu_stats.log" &
GPU_LOGGER_PID=$!

# ---------------------------------------------------------------------------
# Run the beam-selection script with unbuffered output.
# ---------------------------------------------------------------------------
echo "=== Training started at $(date) ===" | tee -a "${OUTPUT_DIR}/run.log"

python3 -u "${PYTHON_SCRIPT}" --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${OUTPUT_DIR}/run.log"

echo "=== Training finished at $(date) ===" | tee -a "${OUTPUT_DIR}/run.log"
