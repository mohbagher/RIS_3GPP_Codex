#!/bin/bash
# RIS Beam Selection — full GPU experiment launcher
# Submit to Slurm with:  sbatch run_ris_experiment.sh

#SBATCH --job-name=ris_beam_sel
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your@email.ac.uk   # uncomment and set your address

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
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Output   : ${OUTPUT_DIR}"
echo "Started  : $(date)"

# ------------------------------------------------------------------
# Run experiment
# ------------------------------------------------------------------
python3 -u ris_beam_selection_v8_3gpp.py --output_dir "${OUTPUT_DIR}"

echo "Finished : $(date)"
