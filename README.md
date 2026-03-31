# RIS_3GPP_Codex

RIS Beam Selection using 3GPP/ETSI-compliant channel models (CDL, TR 38.901) with
Doppler-aware curriculum training, EWC regularisation, and instance normalisation.
Optimised for NVIDIA A100 GPUs on Stanage HPC (University of Sheffield).

## Quick start

### Interactive test (CPU, ~5 s)

```bash
srun --partition=devel --time=00:10:00 --mem=48G --cpus-per-task=4 --pty \
    python3 -u ris_beam_selection_v8_3gpp.py --test
```

### Batch jobs via sbatch

The repository provides two shell scripts (`.sh`) containing Slurm `#SBATCH`
directives. Submit them with `sbatch`:

| Script | Purpose | Partition | Time |
|---|---|---|---|
| `run_ris_test.sh` | Pipeline validation (CPU, `--test` mode) | `devel` | 10 min |
| `run_ris_experiment.sh` | Full GPU experiment (A100) | `gpu` | 12 h |

```bash
# Quick smoke test
sbatch run_ris_test.sh

# Full experiment
sbatch run_ris_experiment.sh
```

Logs are written to `logs/<jobname>_<jobid>.out` / `.err`.
Results are written to `results/<jobname>_<jobid>/`.

## Repository structure

| File | Description |
|---|---|
| `ris_beam_selection_v8_3gpp.py` | Main training and evaluation script |
| `cdl_38901_ris.py` | 3GPP TR 38.901 CDL channel model module |
| `run_ris_experiment.sh` | Slurm shell script for full GPU experiment |
| `run_ris_test.sh` | Slurm shell script for quick CPU validation |