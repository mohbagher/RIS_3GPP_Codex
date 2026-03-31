#!/bin/bash
# ==============================================================
#  Hardened Slurm batch script — RIS Beam Selection v8
#  Stanage HPC (University of Sheffield)
#
#  Usage:
#    sbatch submit_ris_experiment.sh                  # full run
#    sbatch --export=ALL,TEST=1 submit_ris_experiment.sh  # smoke test
# ==============================================================

# ── Resource allocation ────────────────────────────────────────
#SBATCH --job-name=ris_beam_v8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1          # one NVIDIA A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00            # 8 h — enough for 750 k samples / 500 ep

# ── I/O ────────────────────────────────────────────────────────
#SBATCH --output=logs/%x_%j.out   # stdout  → logs/ris_beam_v8_<jobid>.out
#SBATCH --error=logs/%x_%j.err    # stderr  → logs/ris_beam_v8_<jobid>.err

# ── Email (optional — remove or edit address) ──────────────────
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your@email.ac.uk

set -euo pipefail

# ── Environment ────────────────────────────────────────────────
module load Anaconda3/2022.10
source activate ris_env          # adjust env name if different

# ── Output directory ───────────────────────────────────────────
OUTDIR="${SLURM_SUBMIT_DIR}/results/${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}" logs

echo "Job  : ${SLURM_JOB_ID}"
echo "Node : ${SLURMD_NODENAME}"
echo "Out  : ${OUTDIR}"
nvidia-smi

# ── Launch ────────────────────────────────────────────────────
SCRIPT="${SLURM_SUBMIT_DIR}/ris_beam_selection_v8_3gpp.py"

if [[ "${TEST:-0}" == "1" ]]; then
    python -u "${SCRIPT}" --test --output_dir "${OUTDIR}"
else
    python -u "${SCRIPT}" --output_dir "${OUTDIR}"
fi
