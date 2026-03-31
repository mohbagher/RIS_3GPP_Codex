# Running on HPC (Slurm / A100 GPU)

This guide explains how to submit and monitor the `ris_beam_selection_v8_3gpp.py`
experiment on a Slurm-managed HPC cluster using the provided batch script.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Slurm scheduler | Available on the target cluster |
| `gpu` partition & QOS | A100 GPU nodes must be accessible |
| Anaconda module | Loaded via `module load anaconda3` |
| `ris_env` conda environment | Must be created beforehand (see below) |

### Create the conda environment (first time only)

```bash
module load anaconda3
conda create -n ris_env python=3.10 -y
conda activate ris_env
pip install -r requirements.txt   # if a requirements file is present
```

---

## Submitting the job

Run the following command from the **repository root**:

```bash
sbatch run_ris_gpu.sbatch
```

The script resolves the Python file as `Claude_Promo_Period/ris_beam_selection_v8_3gpp.py`
relative to the working directory, so you must submit from the repo root.

To customise resources (time limit, memory, etc.) edit the `#SBATCH` directives
near the top of `run_ris_gpu.sbatch` before submitting.

### Email notifications

Uncomment and fill in the two `##SBATCH --mail-*` lines in `run_ris_gpu.sbatch`
to receive an email when the job starts, ends, or fails:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com
```

---

## Monitoring the job

| Task | Command |
|---|---|
| Check job status | `squeue -u $USER` |
| Watch live log | `tail -f ris_job_<jobid>.out` |
| Watch GPU stats | `tail -f output_<jobid>/gpu_stats.log` |
| Cancel the job | `scancel <jobid>` |

Replace `<jobid>` with the numeric job ID printed by `sbatch` (e.g. `12345`).

---

## Output files

All outputs are written to `output_<jobid>/` in the directory where you ran
`sbatch`:

| File | Contents |
|---|---|
| `output_<jobid>/run.log` | Combined stdout/stderr from the Python script with start/end timestamps and exit code |
| `output_<jobid>/gpu_stats.log` | GPU utilisation, memory, temperature, and power sampled every 30 s |
| `output_<jobid>/ris_beam_selection_v8_3gpp.py` | Snapshot of the Python script used for this run |
| `output_<jobid>/run_ris_gpu.sbatch` | Snapshot of the batch script used for this run |
| `ris_job_<jobid>.out` | Slurm stdout (metadata echoed at job start) |
| `ris_job_<jobid>.err` | Slurm stderr |

Model checkpoints and any other files created by the Python script are also
written inside `output_<jobid>/` because `--output_dir` is passed automatically.

---

## Troubleshooting

**Job fails immediately with "Python script not found"**  
Make sure you are running `sbatch` from the repository root and that
`Claude_Promo_Period/ris_beam_selection_v8_3gpp.py` exists.

**`conda activate` fails**  
Ensure the `anaconda3` module is available (`module avail anaconda`) and that
the `ris_env` environment has been created as described above.

**Out-of-memory errors**  
Increase `--mem` in `run_ris_gpu.sbatch` or reduce the batch size in the
Python script.
