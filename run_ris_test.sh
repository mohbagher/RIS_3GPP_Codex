#!/bin/bash
# RIS Beam Selection — quick CPU pipeline validation (test mode)
# Submit to Slurm with:  sbatch run_ris_test.sh

#SBATCH --job-name=ris_test
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------
module purge
module load Anaconda3
source activate ris_env

OUTPUT_DIR="results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p logs "${OUTPUT_DIR}"

echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Output   : ${OUTPUT_DIR}"
echo "Started  : $(date)"

# ------------------------------------------------------------------
# Run pipeline validation (small parameters, CPU only)
# ------------------------------------------------------------------
python3 -u ris_beam_selection_v8_3gpp.py --test --output_dir "${OUTPUT_DIR}"

echo "Finished : $(date)"
