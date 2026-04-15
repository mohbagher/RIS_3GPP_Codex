# RIS_3GPP_Codex

GPU-accelerated RIS beam-selection pipeline (v8) aligned with 3GPP TR 38.901 /
ETSI GR RIS 003, targeting NVIDIA A100 on the Stanage HPC cluster.

---

## Quick start (smoke test — CPU, ~5 s)

```bash
python ris_beam_selection_v8_3gpp.py --test
```

## Full experiment (batch job — GPU recommended)

```bash
sbatch submit_ris_experiment.sh
```

Outputs land in `results/<job_id>/`.  Logs in `logs/`.

---

## Interactive `srun` vs hardened `sbatch` — behavior and risks

The table below compares running the script via an **interactive `srun`
session** (as commonly tried during development) against submitting it as a
**`sbatch` batch job** using the provided `submit_ris_experiment.sh`.

| Aspect | Interactive `srun` (development) | Hardened `sbatch` (recommended) |
|---|---|---|
| **Partition** | `devel` (strict 10-min wall-clock cap) | `gpu` (up to 8 h or more) |
| **GPU** | None requested → runs on CPU | `--gres=gpu:a100:1` → full A100 |
| **Training speed** | ~100–1 000× slower on CPU; 750 k samples / 500 epochs **will not finish** in 10 min | Typical full run completes in 2–6 h on A100 |
| **AMP / bfloat16** | Silently disabled (PyTorch warns, then degrades to FP32 on CPU) | Fully active; ~2× throughput gain |
| **Memory** | `--mem=48G` shared RAM only — no GPU VRAM; large GPU-resident tensors fall back to CPU | 80 GB RAM + 40 GB HBM; all tensors stay on device |
| **Session persistence** | Job dies if SSH connection drops or devel time-limit is hit; no checkpoint recovery | Detached batch job; survives network disconnects |
| **Stdout / stderr** | Printed to terminal only — lost on disconnect | Captured to `logs/<job>_<id>.{out,err}` |
| **Output files** | Written to CWD (risk of cluttering login node) | Written to `results/<job_id>/` via `--output_dir` |
| **Reproducibility** | Depends on which login node you land on | Fixed node assignment by Slurm |

### Specific risks when using `srun --partition=devel --time=00:10:00`

1. **Job killed at 10 minutes** — the `devel` partition enforces its wall-clock
   limit unconditionally. A full run (750 k samples, 500 + 120 epochs) needs
   hours, not minutes. The process is `SIGKILL`ed mid-training with no
   checkpoint written.

2. **No GPU allocated** — without `--gres=gpu:...`, Slurm does not assign any
   GPU. `torch.cuda.is_available()` returns `False`; the script falls back to
   CPU. AMP (`bfloat16`) and the GradScaler are disabled with a warning.
   Training throughput drops by 2–3 orders of magnitude.

3. **Silent AMP/GradScaler warnings become errors at scale** — on CPU the
   `torch.cuda.amp.GradScaler` is disabled, and `torch.autocast` for
   `device_type='cuda'` is also disabled. These are harmless in `--test` mode
   but indicate the hardware pipeline is not exercised at all.

4. **Results lost on disconnect** — interactive sessions with `--pty` tie
   stdout to your terminal. A dropped SSH connection terminates the process and
   deletes all in-memory state; no model checkpoint or plot is saved.

5. **Login-node pollution** — even a short `srun` on `devel` can run on a
   shared login node if the scheduler routes it there, violating cluster fair-use
   policy.

### When interactive `srun` is appropriate

Use `srun` only for the **`--test` smoke-test** (< 5 s, CPU, 500 samples):

```bash
srun --partition=devel --time=00:05:00 --mem=8G --cpus-per-task=2 \
     python ris_beam_selection_v8_3gpp.py --test
```

For any run without `--test`, use `sbatch submit_ris_experiment.sh`.

---

## Files

| File | Description |
|---|---|
| `ris_beam_selection_v8_3gpp.py` | Main training + evaluation script |
| `cdl_38901_ris.py` | 3GPP TR 38.901 CDL channel model |
| `submit_ris_experiment.sh` | Hardened Slurm batch submission script |