# -*- coding: utf-8 -*-
"""
RIS Beam Selection v8 — 3GPP/ETSI Channel Model + Instance Normalisation
GPU-Optimised for NVIDIA A100 (Stanage HPC, University of Sheffield)

======================================================================
  CHANNEL MODEL (v7 — 3GPP/ETSI compliant)
======================================================================
  [3GPP-1]  Path loss: 3GPP TR 38.901 UMi-Street Canyon
            BS→RIS link (LOS):  PL = 32.4 + 21·log₁₀(d) + 20·log₁₀(fc)
            RIS→UE link (NLOS): PL = 35.3·log₁₀(d) + 22.4 + 21.3·log₁₀(fc)
                                     − 0.3·(h_UE − 1.5)
            Shadow fading: σ_SF = 4.0 dB (LOS), 7.82 dB (NLOS)
  [3GPP-2]  RIS aperture gain per ETSI GR RIS 003:
            G_RIS = N_RIS · (4π·dx·dy / λ²) · A²
            where dx = dy = λ/2 (half-wavelength spacing),
            A = RIS element reflection amplitude
  [3GPP-3]  Thermal noise floor: N₀ = k_B · T_sys · BW · NF
            (constant per hardware — not signal-dependent)
  [3GPP-4]  Transmit power in dBm, full link budget in SI units
  [3GPP-5]  Per-sample log-normal shadow fading for SNR diversity

  Design: The channel is generated with unit-amplitude steering
  vectors (no STEERING_SCALE), then the full link budget
  (P_TX, path loss, RIS gain, shadow fading, thermal noise)
  is applied to the received probes. This keeps the cascade
  channel h normalised for stable codebook operations, while
  the received SNR is physically correct.

======================================================================
  ML PIPELINE (unchanged from v6)
======================================================================
  [FIX-1] 3-way train/val/test split, dual-metric validation
  [FIX-2] Dedup-aware 2D Gaussian label smoothing
  [FIX-3] Multi-mask evaluation (N_EVAL_MASKS seeds)
  [FIX-4] Codebook orthogonality diagnostic
  [SIGMA-FIX] Phase 1 sigma annealing uses PHASE1_MAX_EPOCHS denominator

======================================================================
  DOPPLER (unchanged from v6)
======================================================================
  [D-1] AR(1) channel aging with Clarke's J₀ autocorrelation
  [D-2] Phase 2 curriculum training: warmup → aging probability ramp
  [D-3] EWC regularisation to protect Phase 1 knowledge
  [D-4] SupCon disabled on aged batches
  [D-5] Relabelling: aged batches get oracle labels from aged channel
  [D-6] Physics-correct coherence-time overhead model
  [D-7] Doppler evaluation across speed scenarios
"""

# ==============================================================
# CELL 1: CONFIGURATION — Change parameters HERE ONLY
# ==============================================================
import numpy as np

# -- 1. Array / Codebook Dimensions ----------------------------
N_ROWS = 16
N_COLS = 32
V_ROWS = N_ROWS
V_COLS = N_COLS

# -- 2. Carrier Frequency --------------------------------------
FC_HZ = 28e9

# -- 3. Discrete RIS Phase Configuration -----------------------
# 0 = continuous, 1 = 1-bit {0,pi}, 2 = 2-bit {0,pi/2,pi,3pi/2}
RIS_BITS = 1

# -- 4. Probe Budget -------------------------------------------
BUDGET_LIST_RAW = [32, 64, 128, 256, 512]

# -- 5. Dataset ------------------------------------------------
NUM_SAMPLES = 750_000
TEST_FRAC = 0.2
VAL_FRAC  = 0.1          # [FIX-1] validation split for early stopping
RANDOM_SEED = 42

# -- 5b. Evaluation robustness --------------------------------
N_EVAL_MASKS = 10         # [FIX-3] number of random mask seeds

# -- 6. ETSI RIS Deployment Geometry ----------------------------
#   Scenario: BS on building wall/pole → RIS on facade → UE at street level
#   Direct BS→UE link is blocked (NLOS motivation for RIS)
#   Path loss: ETSI GR RIS 003 far-field cascaded formula
D_BS_RIS_MIN   = 20.0      # m — BS-to-RIS distance range
D_BS_RIS_MAX   = 100.0     # m
D_RIS_UE_MIN   = 8.0       # m — RIS-to-UE distance range (Fix: Fraunhofer near-field boundary)
D_RIS_UE_MAX   = 20.0      # m

# -- 7. RIS Element Properties (ETSI GR RIS 003) ---------------
RIS_ELEM_AMPLITUDE = 0.8       # reflection coefficient amplitude A (1-bit practical value)
# RIS element size: λ/2 × λ/2 (half-wavelength spacing, standard)
# Computed below from FC_HZ

# -- 8. Transmit Power & Receiver (3GPP-standard values) -------
P_TX_DBM        = 30.0       # dBm — BS transmit power (1 W, 3GPP UMi typical)
K_BOLTZMANN     = 1.38e-23   # J/K — Boltzmann's constant
T_SYS_K         = 290.0      # K   — system noise temperature
# [PHYSICS FIX] A 2.5us narrowband probe has an effective measurement BW of 1/T_slot (400 kHz).
BW_HZ           = 1.0 / 2.5e-6 # Hz  — receiver bandwidth for 2.5µs slot
NOISE_FIGURE_DB = 7.0        # dB  — UE receiver noise figure

# -- 8b. Shadow Fading Standard Deviations ---------------------
#   Combined shadow fading for the cascaded link.
#   Reference: ETSI GR RIS 003 recommends log-normal SF for each sub-link.
#   BS→RIS (LOS): σ = 4.0 dB per 3GPP TR 38.901 UMi-SC LOS
#   RIS→UE (NLOS): σ = 7.82 dB per 3GPP TR 38.901 UMi-SC NLOS
#   Combined (independent): σ_total = √(4² + 7.82²) ≈ 8.8 dB
SIGMA_SF_COMBINED_DB = 8.8   # dB — combined shadow fading for cascaded RIS link

# -- 9. 3GPP CDL (Small-Scale Fading) ---------------------------
CDL_C_ASD_DEG = 2.0
CDL_C_ASA_DEG = 15.0
CDL_C_ZSD_DEG = 3.0
CDL_C_ZSA_DEG = 7.0
CDL_N_RAYS    = 20
CDL_CHUNK_SIZE = 256

TEST_SNR_DB     = 10.0   # used ONLY in achievable rate formula

# -- 10. Model Architecture ------------------------------------
HIDDEN = 2048
N_RESBLOCKS = 7
DROPOUT_RATE = 0.15
LEAKY_SLOPE = 0.2

# -- 11. Training Hyperparameters (Phase 1 — static) -----------
BATCH = 16384
PHASE1_MAX_EPOCHS = 500
LR_PATIENCE = 6
PATIENCE    = 25        # Phase 1 early stopping patience (on val_acc)
LR = 8e-3
WEIGHT_DECAY = 1e-4
LR_FACTOR = 0.5

# -- 11b. Phase 2 — Doppler curriculum -------------------------
PHASE2_MAX_EPOCHS      = 120
PHASE2_PATIENCE        = 30
PHASE2_LR_FACTOR       = 0.10
AGING_MAX_SPEED        = 10.0       # m/s — max curriculum speed
PHASE2_WARMUP_EPOCHS   = 5
PHASE2_AGING_PROB_START = 0.10
PHASE2_AGING_PROB_END   = 0.40
PHASE2_AGING_RAMP_FRAC  = 0.80
CURRICULUM_SPEED_START  = 0.5       # m/s
CURRICULUM_RAMP_FRAC    = 0.85

# -- 11c. EWC regularisation -----------------------------------
EWC_LAMBDA       = 5000.0
EWC_FISHER_BATCHES = 60

# -- 12. Label Smoothing (2D Gaussian) -------------------------
SIGMA_START = 2.0
SIGMA_END = 0.5

# -- 13. Loss Function -----------------------------------------
CE_LOSS_WEIGHT  = 0.1
SUPCON_WEIGHT   = 0.5
SUPCON_TEMP     = 0.07
PROJ_DIM        = 128

# -- 14. Evaluation --------------------------------------------
EVAL_BATCH_SIZE = 4096
ONLINE_DEMO_SAMPLES = 200
TOP_K_LIST = [1, 3, 5]

# -- 15. Validation Thresholds ---------------------------------
COV_PASS_THRESHOLD  = 0.45
COV_WARN_THRESHOLD  = 0.30
CORR_PASS_THRESHOLD = 0.70
DIVERSITY_MIN_FRAC  = 0.50

# -- 16. Doppler / Coherence -----------------------------------
T_SLOT_FIXED_S = 2.5e-6        # slot duration for probing overhead
RHO_FLOOR      = 0.15          # minimum ρ floor

SPEED_SCENARIOS = {
    'static':     0.0,
    'pedestrian': 1.0,
    'cyclist':    5.0,
    'vehicular':  30.0 / 3.6,   # ~8.33 m/s
}


# ==============================================================
# CELL 2: IMPORTS AND DEVICE SETUP
# ==============================================================
from cdl_38901_ris import CDL38901RIS, CDLConfig
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import datetime
import argparse

# -- Argument parser --
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='.', help='Directory for job outputs')
parser.add_argument('--test', action='store_true', help='Quick CPU test with small parameters')
args, unknown = parser.parse_known_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if args.test:
    # Reduce array/codebook dimensions so channel generation completes quickly on CPU.
    # N_ROWS/N_COLS must be overridden before the derived N, V, INPUT_DIM constants below.
    # NOTE: test mode is for pipeline validation only (smoke test); it does not
    # verify model quality or production-scale accuracy.
    N_ROWS              = 4
    N_COLS              = 4
    V_ROWS              = N_ROWS
    V_COLS              = N_COLS
    CDL_CHUNK_SIZE      = 64
    BUDGET_LIST_RAW     = [4, 8, 16]
    NUM_SAMPLES         = 500
    TEST_FRAC           = 0.2
    VAL_FRAC            = 0.1
    HIDDEN              = 64
    N_RESBLOCKS         = 1
    BATCH               = 128
    PHASE1_MAX_EPOCHS   = 2
    PATIENCE            = 2
    LR_PATIENCE         = 1
    PHASE2_MAX_EPOCHS   = 2
    PHASE2_PATIENCE     = 2
    PHASE2_WARMUP_EPOCHS = 1
    EWC_FISHER_BATCHES  = 2
    N_EVAL_MASKS        = 1
    ONLINE_DEMO_SAMPLES = 5
    MAX_EPOCHS = PHASE1_MAX_EPOCHS + PHASE2_MAX_EPOCHS
    print("\n" + "=" * 60)
    print("  TEST MODE — pipeline validation only (not model quality)")
    print(f"  NUM_SAMPLES={NUM_SAMPLES}  HIDDEN={HIDDEN}  "
          f"RESBLOCKS={N_RESBLOCKS}  P1={PHASE1_MAX_EPOCHS}  P2={PHASE2_MAX_EPOCHS}")
    print(f"  RIS={N_ROWS}x{N_COLS}  CDL_CHUNK={CDL_CHUNK_SIZE}")
    print("=" * 60 + "\n")


def get_save_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


script_start_time = time.time()

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    try:
        print(f"  Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except AttributeError:
        print(f"  Memory : (unable to query)")
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.set_float32_matmul_precision('high')
print("  TF32 precision  : ENABLED")
print("  AMP bfloat16    : will be activated in training loop")

# -- Derived Constants -----------------------------------------
_C_LIGHT = 3e8
N = N_ROWS * N_COLS
V = V_ROWS * V_COLS
INPUT_DIM = V + 1
WAVELENGTH  = _C_LIGHT / FC_HZ
K_WAVE      = 2 * np.pi / WAVELENGTH
ELEM_SPACING = WAVELENGTH / 2
SNR_LINEAR  = 10 ** (TEST_SNR_DB / 10)
BUDGET_LIST = [min(m, V) for m in BUDGET_LIST_RAW]
MAX_EPOCHS  = PHASE1_MAX_EPOCHS + PHASE2_MAX_EPOCHS

# -- ETSI GR RIS 003 Link Budget (computed from configuration) --
_FC_GHZ = FC_HZ / 1e9

# RIS element dimensions (half-wavelength spacing)
RIS_DX = WAVELENGTH / 2
RIS_DY = WAVELENGTH / 2

# ETSI GR RIS 003 Far-Field Path Loss for RIS-Assisted Link:
#
#   PL_RIS = (4π d₁ d₂)² / (N_RIS · dx · dy · A)²
#
# This is a SINGLE formula for the entire cascaded BS→RIS→UE path.
# It includes: free-space propagation on both sub-links, RIS element
# scattering gain (G_s = 4π dx dy / λ²), and N² coherent beamforming gain.
# Reference: ETSI GR RIS 003 V1.1.1, confirmed by MathWorks 5G Toolbox.
#
# NOTE: Do NOT sum two independent 3GPP path losses — that double-counts
# the free-space loss and misses the coherent array gain structure.

def etsi_ris_path_loss_db(d1, d2, n_ris, dx, dy, a):
    """ETSI GR RIS 003 far-field path loss [dB] for cascaded RIS link."""
    pl_linear = ((4 * np.pi * d1 * d2) / (n_ris * dx * dy * a)) ** 2
    return 10 * np.log10(pl_linear)

# Mean path loss at mean distances
_D_BS_RIS_MEAN = (D_BS_RIS_MIN + D_BS_RIS_MAX) / 2
_D_RIS_UE_MEAN = (D_RIS_UE_MIN + D_RIS_UE_MAX) / 2
_PL_RIS_MEAN_DB = etsi_ris_path_loss_db(
    _D_BS_RIS_MEAN, _D_RIS_UE_MEAN, N, RIS_DX, RIS_DY, RIS_ELEM_AMPLITUDE
)

# Thermal noise power
P_TX_W   = 10 ** ((P_TX_DBM - 30) / 10)           # Watts
N0_W     = K_BOLTZMANN * T_SYS_K * BW_HZ * (10 ** (NOISE_FIGURE_DB / 10))  # Watts
N0_DBM   = 10 * np.log10(N0_W) + 30                # dBm

# Noise standard deviation per complex dimension (for additive noise)
NOISE_STD = float(np.sqrt(N0_W / 2))  # σ per real/imag component

# Expected SNR at mean geometry (reference only — actual varies per sample)
_RX_POWER_DBM = P_TX_DBM - _PL_RIS_MEAN_DB
_SNR_MEAN_DB  = _RX_POWER_DBM - N0_DBM

PHASE_ALPHABET_MAP = {
    0: (None,  None,                                "continuous"),
    1: (2,     [0.0, np.pi],                        "1-bit (2-state)"),
    2: (4,     [0.0, np.pi/2, np.pi, 3*np.pi/2],   "2-bit (4-state)"),
}
if RIS_BITS not in PHASE_ALPHABET_MAP:
    raise ValueError(f"RIS_BITS must be 0, 1, or 2. Got {RIS_BITS}")

N_STATES, phase_list, BITS_LABEL = PHASE_ALPHABET_MAP[RIS_BITS]
PHASE_SET = (
    torch.tensor(phase_list, dtype=torch.float32, device=device)
    if phase_list is not None else None
)

print(f"\n{'=' * 70}")
print(f"  RIS Beam Selection v8 — ETSI/3GPP + Instance Normalisation")
print(f"{'=' * 70}")
print(f"  Array    : {N_ROWS} x {N_COLS} = {N} elements")
print(f"  Codebook : {V_ROWS} x {V_COLS} = {V} beams")
print(f"  Carrier  : {FC_HZ/1e9:.1f} GHz   λ = {WAVELENGTH*1e3:.2f} mm")
print(f"  RIS_BITS : {RIS_BITS} ({BITS_LABEL})")
print(f"  Phase 1  : {PHASE1_MAX_EPOCHS} ep static (patience={PATIENCE} on val_acc)")
print(f"  Phase 2  : {PHASE2_MAX_EPOCHS} ep curriculum (patience={PHASE2_PATIENCE})")
print(f"  --- ETSI GR RIS 003 Link Budget ---")
print(f"  P_TX           = {P_TX_DBM:.1f} dBm")
print(f"  BS→RIS (d₁)    : d ∈ [{D_BS_RIS_MIN:.0f}, {D_BS_RIS_MAX:.0f}] m")
print(f"  RIS→UE (d₂)    : d ∈ [{D_RIS_UE_MIN:.0f}, {D_RIS_UE_MAX:.0f}] m")
print(f"  RIS elements   = {N}  (dx=dy=λ/2={RIS_DX*1e3:.2f}mm, A={RIS_ELEM_AMPLITUDE})")
print(f"  PL formula     : (4π d₁ d₂ / (N·dx·dy·A))²")
print(f"  PL(mean)       = {_PL_RIS_MEAN_DB:.1f} dB  (d₁={_D_BS_RIS_MEAN:.0f}m, d₂={_D_RIS_UE_MEAN:.0f}m)")
print(f"  Shadow fading  : σ_SF = {SIGMA_SF_COMBINED_DB} dB (combined LOS+NLOS)")
print(f"  Noise floor    = {N0_DBM:.1f} dBm  (T={T_SYS_K:.0f}K, BW={BW_HZ/1e6:.0f}MHz, NF={NOISE_FIGURE_DB}dB)")
print(f"  Expected SNR   ≈ {_SNR_MEAN_DB:.1f} dB  (at mean geometry)")
print(f"  Output   : {os.path.abspath(OUTPUT_DIR)}")
print(f"{'=' * 70}")


# ==============================================================
# CELL 3: HELPER FUNCTIONS
# ==============================================================
_row_idx = torch.arange(N_ROWS, device=device).float().repeat_interleave(N_COLS)
_col_idx = torch.arange(N_COLS, device=device).float().repeat(N_ROWS)
ELEM_Y = ((_col_idx - (N_COLS - 1) / 2.0) * ELEM_SPACING).unsqueeze(0)
ELEM_Z = ((_row_idx - (N_ROWS - 1) / 2.0) * ELEM_SPACING).unsqueeze(0)


def steering_vector(azimuth, elevation):
    """
    Unit-amplitude array steering vector for a UPA.

    Returns shape (S, N) complex tensor where N = N_ROWS * N_COLS.
    The steering vector captures spatial phase structure only —
    all amplitude/power effects (path loss, RIS gain, etc.) are
    applied externally via the 3GPP link budget.
    """
    phase = K_WAVE * (
        ELEM_Y * torch.sin(azimuth) * torch.cos(elevation)
        + ELEM_Z * torch.sin(elevation)
    )
    return torch.exp(1j * phase)


# -- ETSI GR RIS 003 Path Loss Function -------------------------
def compute_per_sample_path_loss_db(d_bs_ris, d_ris_ue, dev):
    """
    Compute per-sample RIS cascaded path loss [dB] using ETSI GR RIS 003.

    PL_RIS = (4π d₁ d₂ / (N_RIS · dx · dy · A))²  [linear]

    Plus log-normal shadow fading with combined σ_SF.

    Returns:
        pl_total_db : (S,) tensor — total path loss per sample in dB
    """
    S = d_bs_ris.shape[0]
    d1 = d_bs_ris.squeeze(1)    # (S,)
    d2 = d_ris_ue.squeeze(1)    # (S,)

    # ETSI far-field deterministic path loss
    numerator   = 4 * np.pi * d1 * d2
    denominator = N * RIS_DX * RIS_DY * RIS_ELEM_AMPLITUDE
    pl_db = 20 * torch.log10(numerator / denominator)  # 10*log10(x²) = 20*log10(x)

    # Combined shadow fading (log-normal, independent per realisation)
    sf_db = torch.randn(S, device=dev) * SIGMA_SF_COMBINED_DB

    return pl_db + sf_db


def quantise_phases(phase_tensor, phase_set):
    p = phase_tensor.unsqueeze(-1)
    diff = torch.abs(p - phase_set)
    diff = torch.min(diff, 2 * np.pi - diff)
    idx = diff.argmin(dim=-1)
    return phase_set[idx]


def beam_power(h_channel, beam_indices, codebook):
    beams = codebook[beam_indices.to(codebook.device)]
    h = h_channel.to(codebook.device)
    return (h * beams.conj()).sum(dim=1).abs().pow(2)


def achievable_rate(power, snr_linear):
    return torch.log2(1.0 + snr_linear * power)


def top_k_accuracy(logits, labels, k):
    _, top_k_preds = logits.topk(k, dim=1)
    correct = top_k_preds.eq(labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


# -- Clarke's autocorrelation (Doppler) ------------------------
try:
    from scipy.special import j0 as _scipy_j0
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def clarke_rho_raw(fd, dt):
    """Raw J₀(2πf_D·Δt) — can be negative."""
    if fd < 1e-6 or abs(dt) < 1e-15:
        return 1.0
    x = 2 * np.pi * fd * dt
    if _HAS_SCIPY:
        return float(_scipy_j0(x))
    x2 = x * x
    if abs(x) < 8:
        return float(1 - x2/4 + x2**2/64 - x2**3/2304 + x2**4/147456)
    return float(np.sqrt(2 / (np.pi * abs(x))) * np.cos(abs(x) - np.pi / 4))


def clarke_rho(fd, dt, floor=RHO_FLOOR):
    """Effective ρ = max(|J₀(2πf_D·Δt)|, floor). Static → 1.0."""
    if fd < 1e-6:
        return 1.0
    return max(abs(clarke_rho_raw(fd, dt)), floor)


def clarke_rho_tensor(fd_vec, dt_vec, floor=RHO_FLOOR):
    """Batched ρ computation for training (Taylor expansion)."""
    x = 2 * np.pi * fd_vec * dt_vec
    x2 = x * x
    # [FIX] 5th-order Taylor diverges wildly for x > 3 (vehicular speeds).
    # Use fallback cosine envelope for large arguments to prevent NaN explosions.
    rho_taylor = 1 - x2 / 4 + x2.pow(2) / 64 - x2.pow(3) / 2304 + x2.pow(4) / 147456 - x2.pow(5) / 14745600
    rho_asymp = torch.sqrt(2 / (np.pi * x.abs() + 1e-9)) * torch.cos(x.abs() - np.pi / 4)
    rho = torch.where(x.abs() < 2.5, rho_taylor, rho_asymp)

    rho = rho.abs().clamp(min=floor)
    rho = torch.where(fd_vec < 1.0, torch.ones_like(rho), rho)
    return rho.clamp(max=1.0)


def age_channel_ar1(h, rho, dev):
    """AR(1) channel aging: h_aged = ρ·h + √(1−ρ²)·w."""
    if rho >= 1 - 1e-9:
        return h
    rho = min(rho, 1.0)
    hp = (h.abs() ** 2).mean(1, keepdim=True)
    ns = torch.sqrt(hp / 2)
    w = ns * (torch.randn_like(h.real) + 1j * torch.randn_like(h.real))
    return rho * h + np.sqrt(1 - rho ** 2) * w


def age_channel_ar1_batched(h, rho_vec, dev):
    """Batched AR(1) with per-sample ρ."""
    rc = rho_vec.view(-1, 1).clamp(0, 1)
    hp = (h.abs() ** 2).mean(1, keepdim=True)
    ns = torch.sqrt(hp / 2)
    w = ns * (torch.randn_like(h.real) + 1j * torch.randn_like(h.real))
    return rc * h + torch.sqrt((1 - rc ** 2).clamp(min=0)) * w


# -- Coherence time helpers [D-6] -----------------------------
def get_coherence_time_s(speed_mps):
    """T_c = 0.423 / f_D seconds. Returns inf for static."""
    if speed_mps < 1e-6:
        return float('inf')
    fd = speed_mps * FC_HZ / _C_LIGHT
    return 0.423 / fd


def get_max_feasible_M(speed_mps, t_slot_s):
    """Max probes before probing time exceeds coherence time."""
    tc = get_coherence_time_s(speed_mps)
    if tc == float('inf'):
        return 10_000
    return int(tc / t_slot_s)


print("Helper functions defined (static + Doppler).")

# -- ρ diagnostic printout ------------------------------------
print(f"\n  Clarke's ρ (T_slot={T_SLOT_FIXED_S*1e6:.1f}µs, floor={RHO_FLOOR}):")
for nm, spd in SPEED_SCENARIOS.items():
    fd = spd * FC_HZ / _C_LIGHT if spd > 1e-6 else 0.0
    if fd < 1e-6:
        print(f"    {nm:>12s}: static")
        continue
    tc_s = get_coherence_time_s(spd)
    max_M = get_max_feasible_M(spd, T_SLOT_FIXED_S)
    print(f"    {nm:>12s}: T_c={tc_s*1e3:.3f}ms  max_feasible_M={max_M}")
    for Mc in [32, 64, 128, 256, 453]:
        rm = clarke_rho(fd, Mc / 2 * T_SLOT_FIXED_S)
        re = clarke_rho(fd, Mc * T_SLOT_FIXED_S)
        rr = clarke_rho_raw(fd, Mc * T_SLOT_FIXED_S)
        exceeds = "⚠EXCEEDS T_c" if Mc > max_M else ""
        print(f"      M={Mc:>3}: ρ_mid={rm:.4f} ρ_end={re:.4f} (J₀={rr:+.4f}) {exceeds}")


# ==============================================================
# CELL 4: CODEBOOK GENERATION (with deduplication)
# ==============================================================
def generate_codebook(n_rows, n_cols, v_rows, v_cols, dev,
                      ris_bits=0, phase_set=None):
    nr  = torch.arange(n_rows, device=dev).float()
    nc  = torch.arange(n_cols, device=dev).float()
    uel = 2 * torch.arange(v_rows, device=dev).float() / v_rows - 1
    uaz = 2 * torch.arange(v_cols, device=dev).float() / v_cols - 1
    F_row  = torch.exp(1j * np.pi * torch.outer(nr, uel))
    F_col  = torch.exp(1j * np.pi * torch.outer(nc, uaz))
    F_full = F_row.unsqueeze(1).unsqueeze(3) * F_col.unsqueeze(0).unsqueeze(2)
    cb = F_full.reshape(n_rows * n_cols, v_rows * v_cols).T.contiguous()
    if ris_bits == 0 or phase_set is None:
        return cb
    angles_continuous = torch.angle(cb)
    angles_quantised  = quantise_phases(angles_continuous, phase_set.to(dev))
    return torch.exp(1j * angles_quantised)


def deduplicate_codebook(codebook, ris_bits):
    if ris_bits == 0:
        V_ = codebook.shape[0]
        return codebook, torch.arange(V_, device=codebook.device), V_
    V_, N_ = codebook.shape
    angles = torch.angle(codebook)
    n_states = 2 ** ris_bits
    angle_indices = torch.round(angles / (2 * np.pi / n_states)).long() % n_states
    seen, unique_idx = {}, []
    for v in range(V_):
        key = tuple(angle_indices[v].cpu().tolist())
        if key not in seen:
            seen[key] = v
            unique_idx.append(v)
    unique_idx = torch.tensor(unique_idx, device=codebook.device, dtype=torch.long)
    return codebook[unique_idx], unique_idx, len(unique_idx)


CODEBOOK_RAW = generate_codebook(N_ROWS, N_COLS, V_ROWS, V_COLS, device, RIS_BITS, PHASE_SET)
print(f"Raw codebook shape : {tuple(CODEBOOK_RAW.shape)}")

CODEBOOK, CODEBOOK_UNIQUE_IDX, V_EFF = deduplicate_codebook(CODEBOOK_RAW, RIS_BITS)
print(f"Mode               : {BITS_LABEL}")
print(f"Unique codewords   : {V_EFF} / {V} ({V - V_EFF} duplicates removed)")

V_ORIGINAL = V
V = V_EFF
INPUT_DIM = 2 * V + 1
BUDGET_LIST = [min(m, V) for m in BUDGET_LIST_RAW]

if RIS_BITS > 0:
    cb_cont = generate_codebook(N_ROWS, N_COLS, V_ROWS, V_COLS, device, 0, None)
    cb_cont_sub = cb_cont[CODEBOOK_UNIQUE_IDX]
    mse = (CODEBOOK - cb_cont_sub).abs().pow(2).mean().item()
    theory_loss_dB = -20 * np.log10(np.sinc(1 / 2**RIS_BITS))
    print(f"  Quantisation MSE       : {mse:.4f}")
    print(f"  Theoretical BF loss    : {theory_loss_dB:.2f} dB")
    del cb_cont, cb_cont_sub
del CODEBOOK_RAW

print(f"  V (effective) : {V}")
print(f"  INPUT_DIM     : {INPUT_DIM}")

# ==============================================================
# CELL 5: CHANNEL GENERATION (3GPP TR 38.901 CDL-C compliant)
#
# Replaces old simplified model (LOS + hardcoded NLOS/static powers)
# with a CDL-style geometric cluster/ray synthesis:
#   - Cluster means from 3GPP TR 38.901 Table 7.7.1-3 (CDL-C)
#   - Per-cluster powers from table (normalised)
#   - Per-ray angle offsets from 3GPP CDL ray offset set
#   - Random per-ray phases
#
# Returns:
#   cascade : (S,N) complex channel used by the pipeline
#   h_static: (S,N) diagnostic "quasi-static-like" component (kept for compatibility)
# ==============================================================
# --- 3GPP TR 38.901 CDL-C cluster table (Table 7.7.1-3) ---
# Columns: [delay_norm, power_dB, AOD_deg, AOA_deg, ZOD_deg, ZOA_deg]
# ==============================================================
# CELL 5: CHANNEL GENERATION (3GPP TR 38.901 CDL via module)
# ==============================================================
print(f"Generating {NUM_SAMPLES:,} cascaded channels (CDL module)...")

cdl_cfg = CDLConfig(
    fc_hz=FC_HZ,
    n_rows=N_ROWS,
    n_cols=N_COLS,
    c_asd_deg=CDL_C_ASD_DEG,
    c_asa_deg=CDL_C_ASA_DEG,
    c_zsd_deg=CDL_C_ZSD_DEG,
    c_zsa_deg=CDL_C_ZSA_DEG,
    n_rays=CDL_N_RAYS,
    chunk_size=CDL_CHUNK_SIZE,
    shared_cluster_geometry=False,
    random_global_rotation=True,
)

cdl_gen = CDL38901RIS(cdl_cfg, device=device)

# h: cascade channel (S,N), unit-order amplitude, no path loss/noise here
cascade_channel, h1_link, h2_link = cdl_gen.generate(NUM_SAMPLES, return_debug=False)

print(f"  Shape  : {tuple(cascade_channel.shape)}")
print(f"  Memory : {cascade_channel.element_size() * cascade_channel.nelement() / 1e6:.1f} MB")
print(f"  CDL    : cASD={CDL_C_ASD_DEG}°, cASA={CDL_C_ASA_DEG}°, "
      f"cZSD={CDL_C_ZSD_DEG}°, cZSA={CDL_C_ZSA_DEG}°, rays={CDL_N_RAYS}")


# ==============================================================
# CELL 6: ORACLE LABELLING
# ==============================================================
print(f"Computing oracle beam labels over {V} unique codewords...")
all_beam_powers = torch.abs(torch.matmul(cascade_channel, CODEBOOK.conj().T)) ** 2
oracle_labels   = torch.argmax(all_beam_powers, dim=1)
oracle_powers   = all_beam_powers[torch.arange(NUM_SAMPLES, device=device), oracle_labels]
beam_counts     = torch.bincount(oracle_labels, minlength=V).float()
beams_used      = (beam_counts > 0).sum().item()
max_frac        = beam_counts.max().item() / NUM_SAMPLES * 100
print(f"  Beams used         : {int(beams_used)} / {V} ({beams_used/V*100:.1f}%)")
print(f"  Most popular beam  : {max_frac:.2f}% of samples")
del all_beam_powers
if device.type == 'cuda':
    torch.cuda.empty_cache()


# ==============================================================
# CELL 7: MAGNITUDE PROBING WITH 3GPP LINK BUDGET
#
# Architecture:
#   1. Compute clean received probes: y_clean = PROBE_MATRIX @ h
#   2. Apply per-sample received power: P_rx = P_tx / PL(d₁, d₂, SF)
#   3. Scale probes to physical amplitude: y_phys = √P_rx · y_clean / √(mean_power)
#   4. Add thermal noise: y = y_phys + N(0, σ²_noise)
#   5. Take log-magnitude and normalise as features
# ==============================================================
print(f"Generating {V} magnitude probes with 3GPP link budget...")
torch.manual_seed(RANDOM_SEED)
qpsk_indices = torch.randint(0, 4, (V, N), device=device)
PROBE_MATRIX = torch.exp(1j * qpsk_indices.float() * np.pi / 2)

# Step 1: Clean received probes (unit-power channel × random probes)
received_clean = torch.matmul(cascade_channel, PROBE_MATRIX.T)

# Step 2: Per-sample path loss from random geometry
print("  Computing per-sample 3GPP path loss...")
torch.manual_seed(RANDOM_SEED + 1000)  # independent of channel seed
d_bs_ris = torch.rand(NUM_SAMPLES, 1, device=device) * (D_BS_RIS_MAX - D_BS_RIS_MIN) + D_BS_RIS_MIN
d_ris_ue = torch.rand(NUM_SAMPLES, 1, device=device) * (D_RIS_UE_MAX - D_RIS_UE_MIN) + D_RIS_UE_MIN

pl_total_db = compute_per_sample_path_loss_db(d_bs_ris, d_ris_ue, device)

# Step 3: Exact physical scaling with stabilized tensor operations
pl_linear = 10 ** (pl_total_db / 10)
p_rx_w    = P_TX_W / pl_linear

# To prevent tensor values from exploding or underflowing, we use a safe noise floor multiplier
# 1.0 multiplier = safe mathematical space. The SNR ratio is perfectly maintained.
SAFE_NOISE_FLOOR = 1.0
NOISE_MULTIPLIER = SAFE_NOISE_FLOOR / np.sqrt(N0_W / 2.0)

# Scale received power into "safe tensor space"
p_rx_safe = p_rx_w * (NOISE_MULTIPLIER ** 2)
rx_amplitude = torch.sqrt(p_rx_safe).unsqueeze(1)

clean_power_per_sample = (received_clean.abs() ** 2).mean(dim=1, keepdim=True)
received_physical = rx_amplitude * (received_clean / torch.sqrt(clean_power_per_sample + 1e-30))

# Step 4: Additive thermal noise (scaled to SAFE_NOISE_FLOOR)
meas_noise = SAFE_NOISE_FLOOR * (
    torch.randn_like(received_physical.real) + 1j * torch.randn_like(received_physical.real)
)
received_noisy = received_physical + meas_noise

# Diagnostic: report SNR distribution
signal_power_per_sample = (received_physical.abs() ** 2).mean(dim=1)
noise_power_total = SAFE_NOISE_FLOOR ** 2 * 2  # E[|n|²] for complex noise
snr_per_sample_db = 10 * torch.log10(
    signal_power_per_sample / (noise_power_total + 1e-30)
)
print(f"  [3GPP] Per-sample SNR distribution (from link budget):")
print(f"         Min  = {snr_per_sample_db.min().item():.1f} dB")
print(f"         Max  = {snr_per_sample_db.max().item():.1f} dB")
print(f"         Mean = {snr_per_sample_db.mean().item():.1f} dB")
print(f"         Std  = {snr_per_sample_db.std().item():.1f} dB")
print(f"  [3GPP] Path loss range: {pl_total_db.min().item():.1f} to {pl_total_db.max().item():.1f} dB")
print(f"  [3GPP] σ_noise = {NOISE_STD:.4e} (N₀ = {N0_W:.2e} W)")

log_magnitude  = 10 * torch.log10(received_noisy.abs() + 1e-9)

# ──────────────────────────────────────────────────────────────
# [v8] INSTANCE NORMALISATION — removes absolute power from features
#
# Problem: Global normalisation (subtract dataset mean) preserves
# absolute received power as a feature. The ResNet learns to use
# absolute power as a shortcut for distance, then collapses when
# the SNR robustness test forces uniform power across all samples.
#
# Solution: Subtract each sample's own mean log-magnitude (across
# all V probes). This erases absolute path loss and forces the
# network to learn from the RELATIVE spatial pattern of probe
# measurements — which is the actual beam-discriminative signal.
#
# We still divide by a global std for scale stability across the
# dataset, so gradient magnitudes remain well-behaved.
# ──────────────────────────────────────────────────────────────
SAMPLE_MEANS = log_magnitude.mean(dim=1, keepdim=True)   # (S, 1) — per-sample mean
X_GLOBAL_STD = log_magnitude.std()                        # scalar — global scale

X_full_norm = (log_magnitude - SAMPLE_MEANS) / X_GLOBAL_STD

X_cpu = X_full_norm.cpu().float()
y_cpu = oracle_labels.cpu()

BUDGET_LIST = [min(m, V) for m in BUDGET_LIST_RAW]
print(f"  Feature tensor : {tuple(X_cpu.shape)}")
print(f"  [v8] Instance normalisation: per-sample mean subtracted, global std = {X_GLOBAL_STD.item():.4f} dB")

# Store for use in eval/inference (only the global std is needed —
# the per-sample mean is always recomputed from the probe measurements)
X_STD = X_GLOBAL_STD
# X_MEAN is NOT stored — instance norm recomputes it per sample

# Store path loss data for Phase 2 aging (need per-train-sample PL)
PL_TOTAL_DB_ALL = pl_total_db.clone()  # (NUM_SAMPLES,)

del received_clean, received_physical, signal_power_per_sample, snr_per_sample_db
del meas_noise, received_noisy, log_magnitude, X_full_norm, SAMPLE_MEANS
del d_bs_ris, d_ris_ue, pl_total_db, pl_linear, p_rx_w, rx_amplitude, clean_power_per_sample
if device.type == 'cuda':
    torch.cuda.empty_cache()


# ==============================================================
# CELL 8: DATASET SPLIT — GPU-RESIDENT  [FIX-1: 3-way split]
# ==============================================================
all_indices = np.arange(NUM_SAMPLES)

# First split: train+val vs test
X_trainval_cpu, X_test, y_trainval_cpu, y_test, idx_trainval, idx_test = train_test_split(
    X_cpu, y_cpu, all_indices, test_size=TEST_FRAC, random_state=RANDOM_SEED
)

# Second split: train vs val
val_relative_frac = VAL_FRAC / (1.0 - TEST_FRAC)
X_train_cpu, X_val_cpu, y_train_cpu, y_val_cpu, idx_train, idx_val = train_test_split(
    X_trainval_cpu, y_trainval_cpu, idx_trainval,
    test_size=val_relative_frac, random_state=RANDOM_SEED
)

h_test = cascade_channel[idx_test].cpu()
# [D-5] Keep train channels on GPU for Phase 2 aging/relabelling
h_train_gpu = cascade_channel[idx_train]

# Per-split path loss for link-budget-aware noise in Phase 2 and eval
PL_TRAIN_DB = PL_TOTAL_DB_ALL[idx_train]   # (N_train,) on GPU
PL_TEST_DB  = PL_TOTAL_DB_ALL[idx_test]    # (N_test,) on GPU

X_train_gpu = X_train_cpu.to(device)
y_train_gpu = y_train_cpu.to(device)
X_val_gpu   = X_val_cpu.to(device)
y_val_gpu   = y_val_cpu.to(device)

# DataLoader includes sample indices for Phase 2 channel lookup
train_indices_gpu = torch.arange(len(X_train_gpu), device=device)
train_dataset = TensorDataset(X_train_gpu, y_train_gpu, train_indices_gpu)
train_loader  = DataLoader(
    train_dataset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

INPUT_DIM = 2 * V + 1

print(f"Train      : {len(X_train_gpu):,} samples (GPU-resident)")
print(f"Validation : {len(X_val_gpu):,} samples (GPU-resident)  [FIX-1]")
print(f"Test       : {len(X_test):,} samples (CPU)")
print(f"Input dim  : {INPUT_DIM} ({V} probes + {V} mask + 1 budget)")
print(f"Batches/ep : {len(train_loader)}")

xb_check, yb_check, _ = next(iter(train_loader))
assert xb_check.device.type == 'cuda' if device.type == 'cuda' else True
print(f"Batch device: {xb_check.device}  shape: {tuple(xb_check.shape)}")


# ==============================================================
# CELL 9: VALIDATION AND DIAGNOSTICS
# ==============================================================
print(f"\n{'=' * 65}")
print(f"  PRE-TRAINING VALIDATION AND DIAGNOSTICS")
print(f"{'=' * 65}")

all_pass = True

print("\n[1/8] Channel Amplitude Statistics")
flat_amp = torch.abs(cascade_channel).view(-1)
amp_mean = flat_amp.mean().item()
amp_std  = flat_amp.std().item()
amp_cov  = amp_std / amp_mean
if amp_cov > COV_PASS_THRESHOLD:
    status = "PASS"
elif amp_cov > COV_WARN_THRESHOLD:
    status = "WARN — CoV is low"; all_pass = False
else:
    status = "FAIL — CoV too low"; all_pass = False
print(f"  Mean={amp_mean:.4f}  Std={amp_std:.4f}  CoV={amp_cov:.3f}  [{status}]")

print("\n[2/8] Adjacent-Element Spatial Correlation")
sample_h = cascade_channel[min(500, NUM_SAMPLES - 1)]  # diagnostic sample for correlation check
h_left  = sample_h[:-1]; h_right = sample_h[1:]
corr = (torch.abs(torch.sum(h_left * h_right.conj()))
        / (h_left.abs().norm() * h_right.abs().norm() + 1e-12)).item()
status = "PASS" if corr > CORR_PASS_THRESHOLD else "WARN — low spatial correlation"
if corr <= CORR_PASS_THRESHOLD: all_pass = False
print(f"  Correlation={corr:.4f}  (threshold={CORR_PASS_THRESHOLD})  [{status}]")

print("\n[3/8] Oracle Beam Diversity")
beam_counts_v  = torch.bincount(oracle_labels, minlength=V).float()
beams_used_v   = (beam_counts_v > 0).sum().item()
max_beam_frac  = beam_counts_v.max().item() / NUM_SAMPLES * 100
diversity_frac = beams_used_v / V
status = "PASS" if diversity_frac > DIVERSITY_MIN_FRAC else "WARN — many beams never selected"
if diversity_frac <= DIVERSITY_MIN_FRAC: all_pass = False
print(f"  Beams used: {int(beams_used_v)}/{V} ({diversity_frac*100:.1f}%)  [{status}]")
print(f"  Most popular beam: {max_beam_frac:.2f}%")



print("\n[4/8] Feature Distribution")
feat_mean = X_cpu.mean().item(); feat_std = X_cpu.std().item()
feat_skew = ((X_cpu - feat_mean)**3).mean().item() / (feat_std**3 + 1e-12)
feat_kurt = ((X_cpu - feat_mean)**4).mean().item() / (feat_std**4 + 1e-12) - 3
print(f"  Mean={feat_mean:.4f}  Std={feat_std:.4f}  Skew={feat_skew:.3f}  Kurt={feat_kurt:.3f}")
if abs(feat_skew) < 2.0 and abs(feat_kurt) < 7.0:
    print("  PASS — distribution is well-behaved")
else:
    print("  WARN — distribution may cause training instability"); all_pass = False

print("\n[5/8] Label Imbalance")
label_counts    = torch.bincount(y_cpu, minlength=V).float()
nonzero_labels  = label_counts[label_counts > 0]
label_entropy   = -(nonzero_labels / NUM_SAMPLES * torch.log2(nonzero_labels / NUM_SAMPLES + 1e-12)).sum().item()
max_entropy     = np.log2(V)
entropy_ratio   = label_entropy / max_entropy
max_label_count = label_counts.max().item()
max_label_idx   = label_counts.argmax().item()
max_label_pct   = max_label_count / NUM_SAMPLES * 100
eff_classes     = 2 ** label_entropy
print(f"  Entropy: {label_entropy:.2f}/{max_entropy:.2f} ({entropy_ratio*100:.1f}%)")
print(f"  Effective classes: {eff_classes:.0f}/{V}")
print(f"  Most frequent beam: #{max_label_idx} ({max_label_pct:.1f}%)")
issues = []
if max_label_pct > 10:    issues.append(f"FAIL — beam #{max_label_idx} has {max_label_pct:.1f}% of labels")
if eff_classes < 0.2 * V: issues.append(f"FAIL — effective classes ({eff_classes:.0f}) < 20% of V")
if entropy_ratio < 0.70:  issues.append(f"WARN — entropy ratio {entropy_ratio*100:.1f}% < 70%")
if issues:
    all_pass = False
    for iss in issues: print(f"  {iss}")
else:
    print("  PASS — labels are well-balanced")

print("\n[6/8] Codebook Orthogonality")
torch.manual_seed(RANDOM_SEED)
n_sub   = min(50, V)
sub_idx = torch.randperm(V, device=device)[:n_sub]
cb_sub  = CODEBOOK[sub_idx]
gram      = torch.matmul(cb_sub, cb_sub.conj().T)
gram_norm = gram.abs() / N
mask_diag = ~torch.eye(n_sub, device=device, dtype=torch.bool)
off_diag  = gram_norm[mask_diag]
off_mean  = off_diag.mean().item()
off_max   = off_diag.max().item()
n_identical = (off_diag > 0.999).sum().item()
print(f"  Off-diagonal  mean={off_mean:.4f}  max={off_max:.4f}  identical={n_identical}")
if n_identical > 0:
    print("  FAIL — identical codeword pairs found after deduplication!"); all_pass = False
elif off_max < 0.3:
    print("  PASS — codewords are sufficiently orthogonal")
else:
    print(f"  WARN — max off-diagonal={off_max:.4f}  (near-duplicate beams exist)")
    all_pass = False   # [FIX-4]

print(f"\n[7/8] Codebook Deduplication Status")
print(f"  Original: {V_ORIGINAL}  After dedup: {V}  Removed: {V_ORIGINAL - V}")
if V >= V_ORIGINAL * 0.3:
    print("  PASS")
else:
    print("  WARN — very coarse quantisation"); all_pass = False

print(f"\n[8/8] ρ values and feasibility:")
for nm, spd in SPEED_SCENARIOS.items():
    fd = spd * FC_HZ / _C_LIGHT if spd > 1e-6 else 0.0
    if fd < 1e-6:
        print(f"  {nm}: static")
        continue
    max_M = get_max_feasible_M(spd, T_SLOT_FIXED_S)
    for Mc in [128, 256, 453]:
        re = clarke_rho(fd, Mc * T_SLOT_FIXED_S)
        feas = "✓" if Mc <= max_M else "✗ EXCEEDS T_c"
        print(f"  {nm} M={Mc}: ρ_end={re:.4f} {feas}")

print(f"\n{'=' * 65}")
print("  ALL CHECKS PASSED" if all_pass else "  SOME CHECKS FAILED — review above")
print(f"{'=' * 65}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(flat_amp.cpu().numpy(), bins=100, density=True, alpha=0.7, color='steelblue')
axes[0].set_xlabel('|h|'); axes[0].set_ylabel('Density')
axes[0].set_title(f'Channel Amplitude (CoV={amp_cov:.3f})')
axes[1].bar(range(V), beam_counts_v.cpu().numpy(), width=1.0, color='coral', alpha=0.7)
axes[1].axhline(NUM_SAMPLES * 0.10, color='red', ls='--', alpha=0.5, label='10% threshold')
axes[1].set_xlabel('Beam Index'); axes[1].set_ylabel('Count')
axes[1].set_title(f'Oracle Beam Distribution ({int(beams_used_v)}/{V} used)'); axes[1].legend()
axes[2].hist(X_cpu.numpy().flatten()[::100], bins=100, density=True, alpha=0.7, color='seagreen')
axes[2].set_xlabel('Normalised log|y|'); axes[2].set_ylabel('Density')
axes[2].set_title(f'Feature Distribution (skew={feat_skew:.2f})')
plt.tight_layout()
plt.savefig(get_save_path('Feature_Distribution.png')); plt.close()
print("Saved: Feature_Distribution.png")


# ==============================================================
# CELL 10: MODEL DEFINITION  (+ SupCon Projection Head)
# ==============================================================
class ResBlock(nn.Module):
    """x -> [Linear->BN->LeakyReLU->Dropout->Linear->BN] -> (+x) -> LeakyReLU -> Dropout"""
    def __init__(self, size, dropout_rate=0.0, leaky_slope=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size), nn.BatchNorm1d(size), nn.LeakyReLU(leaky_slope),
            nn.Dropout(dropout_rate),
            nn.Linear(size, size), nn.BatchNorm1d(size),
        )
        self.activation = nn.LeakyReLU(leaky_slope)
        self.dropout    = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.activation(x + self.block(x)))


class RISClassifier(nn.Module):
    """
    Input (2V+1) -> encoder -> ResBlocks -> classifier -> V logits
                                         -> proj_head  -> L2-norm features (for SupCon)
    """
    def __init__(self, input_dim, output_dim, hidden, n_resblocks,
                 dropout_rate=0.0, leaky_slope=0.2, proj_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(leaky_slope),
        )
        self.res_blocks  = nn.Sequential(
            *[ResBlock(hidden, dropout_rate, leaky_slope) for _ in range(n_resblocks)]
        )
        self.classifier = nn.Linear(hidden, output_dim)

        # Projection head for SupCon  (2-layer MLP, L2-normalised output)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x):
        feat = self.res_blocks(self.encoder(x))
        logits = self.classifier(feat)
        return logits

    def forward_with_proj(self, x):
        """Returns (logits, l2-normalised projection) — used during training."""
        feat = self.res_blocks(self.encoder(x))
        logits = self.classifier(feat)
        proj   = F.normalize(self.proj_head(feat), dim=1)
        return logits, proj


# Supervised Contrastive Loss
def supcon_loss(features, labels, temperature=0.07):
    """
    features : (B, proj_dim)  — already L2-normalised
    labels   : (B,)           — integer class indices
    Returns scalar SupCon loss.
    """
    B = features.shape[0]
    device_ = features.device

    sim = torch.matmul(features, features.T) / temperature
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    labels_col = labels.contiguous().view(-1, 1)
    pos_mask   = torch.eq(labels_col, labels_col.T).float().to(device_)
    self_mask  = torch.eye(B, device=device_)
    pos_mask   = (pos_mask - self_mask).clamp(min=0)

    exp_sim    = torch.exp(sim) * (1 - self_mask)
    log_prob   = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    n_pos = pos_mask.sum(dim=1)
    safe_n_pos = n_pos.clamp(min=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / safe_n_pos

    has_pos = (n_pos > 0).float()
    loss    = -(mean_log_prob_pos * has_pos).sum() / (has_pos.sum() + 1e-9)
    return loss


def get_gaussian_labels(labels, v_size, sigma, dev,
                        v_rows=None, v_cols=None, row_map=None, col_map=None):
    """
    2D Gaussian soft labels using spatial distance in the original DFT grid.

    [FIX-2] After codebook deduplication, v_size != v_rows*v_cols, so we
    use precomputed row_map / col_map tensors of shape (v_size,).
    """
    if row_map is not None and col_map is not None:
        r_oracle = row_map[labels].float()
        c_oracle = col_map[labels].float()
        r_all    = row_map.float()
        c_all    = col_map.float()
        dr = torch.abs(r_all.unsqueeze(0) - r_oracle.unsqueeze(1))
        dr = torch.min(dr, v_rows - dr)
        dc = torch.abs(c_all.unsqueeze(0) - c_oracle.unsqueeze(1))
        dc = torch.min(dc, v_cols - dc)
        dist2 = dr**2 + dc**2
    elif v_rows is not None and v_cols is not None and v_rows * v_cols == v_size:
        r_oracle = (labels // v_cols).float()
        c_oracle = (labels %  v_cols).float()
        all_v = torch.arange(v_size, device=dev)
        r_all = (all_v // v_cols).float()
        c_all = (all_v %  v_cols).float()
        dr = torch.abs(r_all.unsqueeze(0) - r_oracle.unsqueeze(1))
        dr = torch.min(dr, v_rows - dr)
        dc = torch.abs(c_all.unsqueeze(0) - c_oracle.unsqueeze(1))
        dc = torch.min(dc, v_cols - dc)
        dist2 = dr**2 + dc**2
    else:
        all_v   = torch.arange(v_size, device=dev).float()
        oracle_f = labels.float()
        diff  = torch.abs(all_v.unsqueeze(0) - oracle_f.unsqueeze(1))
        diff  = torch.min(diff, v_size - diff)
        dist2 = diff**2
    soft = torch.exp(-0.5 * dist2 / sigma**2)
    return soft / soft.sum(dim=1, keepdim=True)


# [FIX-2] Build row/col maps from CODEBOOK_UNIQUE_IDX back into original grid
if V != V_ROWS * V_COLS:
    BEAM_ROW_MAP = (CODEBOOK_UNIQUE_IDX // V_COLS).to(device)
    BEAM_COL_MAP = (CODEBOOK_UNIQUE_IDX %  V_COLS).to(device)
    SMOOTH_V_ROWS = V_ROWS
    SMOOTH_V_COLS = V_COLS
    smooth_mode = "2D Gaussian (remapped after dedup) [FIX-2]"
    print(f"  [FIX-2] Remapped {V} deduplicated beams to original {V_ROWS}x{V_COLS} grid")
else:
    BEAM_ROW_MAP = None
    BEAM_COL_MAP = None
    SMOOTH_V_ROWS = V_ROWS
    SMOOTH_V_COLS = V_COLS
    smooth_mode = "2D Gaussian"
print(f"Label smoothing : {smooth_mode}")

model = RISClassifier(INPUT_DIM, V, HIDDEN, N_RESBLOCKS,
                      DROPOUT_RATE, LEAKY_SLOPE, PROJ_DIM).to(device)

kl_criterion = nn.KLDivLoss(reduction='batchmean')
ce_criterion = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel Architecture:")
print(f"  Input      : {INPUT_DIM}")
print(f"  Hidden     : {HIDDEN}")
print(f"  ResBlocks  : {N_RESBLOCKS}")
print(f"  Dropout    : {DROPOUT_RATE}")
print(f"  Output     : {V} beams")
print(f"  Proj head  : {HIDDEN} -> {HIDDEN} -> {PROJ_DIM}  [SupCon]")
print(f"  Params     : {n_params:,}")


# ==============================================================
# CELL 11a: EWC HELPER FUNCTIONS [D-3]
# ==============================================================
def compute_fisher(model_, dataloader, budget_tensor_, n_batches=60, dev='cuda'):
    """Estimate diagonal Fisher information from Phase 1 model."""
    model_.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model_.named_parameters() if p.requires_grad}
    count = 0
    total_samples = 0
    for xb, yb, _ in dataloader:
        if count >= n_batches:
            break
        B = xb.shape[0]
        bi = torch.randint(len(BUDGET_LIST), (B,), device=dev)
        Mv = budget_tensor_[bi]
        rr = torch.rand(B, V, device=dev).argsort(1)
        mask = (rr < Mv.unsqueeze(1)).float()
        xf = torch.cat([xb * mask, mask, (Mv.float() / V).unsqueeze(1)], 1)

        model_.zero_grad()
        logits = model_(xf)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, yb)
        loss.backward()

        for n, p in model_.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2) * B
        total_samples += B
        count += 1

    for n in fisher:
        fisher[n] /= max(total_samples, 1)

    model_.train()
    return fisher, total_samples


def ewc_penalty(model_, fisher, p1_params, lam):
    """EWC penalty: λ/2 * Σ F_i (θ_i − θ*_i)²"""
    loss = 0.0
    for n, p in model_.named_parameters():
        if n in fisher and p.requires_grad:
            loss += (fisher[n] * (p - p1_params[n]).pow(2)).sum()
    return lam / 2.0 * loss


# ==============================================================
# CELL 11b: PHASE 1 TRAINING — Static + SupCon
# [FIX-1] Early stopping on val_acc, LR scheduling on fixed-σ val KL
# ==============================================================
train_start = time.time()

optimizer_p1 = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_p1, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR
)

best_val_acc = -1.0
best_loss    = float('inf')
best_weights = None
no_improve   = 0

history = {
    'loss': [], 'kl': [], 'ce': [], 'supcon': [], 'ewc': [],
    'lr': [], 'sigma': [],
    'val_loss': [], 'val_loss_fixed': [], 'val_acc': [],
    'aging_pct': [], 'phase': [],
}

scaler = torch.amp.GradScaler('cuda')
budget_tensor = torch.tensor(BUDGET_LIST, device=device, dtype=torch.long)

model.train()
print(f"\n{'=' * 90}")
print(f"  PHASE 1: Static training ({PHASE1_MAX_EPOCHS} ep, patience={PATIENCE} on val_acc)")
print(f"  Model: {HIDDEN}×{N_RESBLOCKS} = {n_params:,} params")
print(f"{'=' * 90}")
print(f"  Sigma annealing : {SIGMA_START} -> {SIGMA_END}")
print(f"  LR scheduler    : ReduceOnPlateau on val_loss (fixed σ={SIGMA_END})")
print(f"  Smoothing       : {smooth_mode}")
print(f"  SupCon weight   : {SUPCON_WEIGHT}  temp={SUPCON_TEMP}")
print()
print(f"{'Epoch':>6} {'sigma':>5} {'Total':>9} {'KL':>9} {'CE':>9} {'SupCon':>9} "
      f"{'Val Acc':>9} {'lr':>9} status")
print("-" * 100)

p1_epoch_count = 0
for epoch in range(1, PHASE1_MAX_EPOCHS + 1):
    epoch_start = time.time()
    # [SIGMA-FIX] Use PHASE1_MAX_EPOCHS, not MAX_EPOCHS (P1+P2)
    frac  = (epoch - 1) / max(PHASE1_MAX_EPOCHS - 1, 1)
    sigma = max(SIGMA_END, SIGMA_START - (SIGMA_START - SIGMA_END) * frac)

    # ── Training pass ──
    model.train()
    epoch_loss_total  = 0.0
    epoch_loss_kl     = 0.0
    epoch_loss_ce     = 0.0
    epoch_loss_supcon = 0.0
    n_batches = 0

    for xb, yb, _ in train_loader:
        B = xb.shape[0]

        budget_idx  = torch.randint(len(BUDGET_LIST), (B,), device=device)
        M_vals      = budget_tensor[budget_idx]
        rand_ranks  = torch.rand(B, V, device=device).argsort(dim=1)
        mask        = (rand_ranks < M_vals.unsqueeze(1)).float()
        masked_xb   = xb * mask
        budget_sc   = (M_vals.float() / V).unsqueeze(1)
        xb_full     = torch.cat([masked_xb, mask, budget_sc], dim=1)

        optimizer_p1.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, proj_feats = model.forward_with_proj(xb_full)
            log_probs    = F.log_softmax(logits, dim=1)
            soft_targets = get_gaussian_labels(
                yb, V, sigma, device,
                v_rows=SMOOTH_V_ROWS, v_cols=SMOOTH_V_COLS,
                row_map=BEAM_ROW_MAP, col_map=BEAM_COL_MAP
            )
            loss_kl     = kl_criterion(log_probs, soft_targets)
            loss_ce     = ce_criterion(logits, yb)
            loss_supcon = supcon_loss(proj_feats.float(), yb, temperature=SUPCON_TEMP)
            loss = loss_kl + CE_LOSS_WEIGHT * loss_ce + SUPCON_WEIGHT * loss_supcon

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_p1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer_p1)
        scaler.update()

        epoch_loss_total  += loss.item()
        epoch_loss_kl     += loss_kl.item()
        epoch_loss_ce     += loss_ce.item()
        epoch_loss_supcon += loss_supcon.item()
        n_batches += 1

    avg_total  = epoch_loss_total  / n_batches
    avg_kl     = epoch_loss_kl     / n_batches
    avg_ce     = epoch_loss_ce     / n_batches
    avg_supcon = epoch_loss_supcon / n_batches
    lr_now     = optimizer_p1.param_groups[0]['lr']

    # ── Validation pass [FIX-1] ──
    model.eval()
    with torch.no_grad():
        val_loss_sum       = 0.0
        val_loss_fixed_sum = 0.0
        val_correct  = 0
        val_n        = 0

        for vi in range(0, len(X_val_gpu), BATCH):
            vb_x_raw = X_val_gpu[vi:vi + BATCH]
            vb_y     = y_val_gpu[vi:vi + BATCH]
            B_vb     = vb_x_raw.shape[0]

            vb_budget_idx = torch.randint(len(BUDGET_LIST), (B_vb,), device=device)
            vb_M_vals     = budget_tensor[vb_budget_idx]
            vb_rand_ranks = torch.rand(B_vb, V, device=device).argsort(dim=1)
            vb_mask       = (vb_rand_ranks < vb_M_vals.unsqueeze(1)).float()
            vb_masked     = vb_x_raw * vb_mask
            vb_budget_sc  = (vb_M_vals.float() / V).unsqueeze(1)
            vb_full       = torch.cat([vb_masked, vb_mask, vb_budget_sc], dim=1)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                v_logits    = model(vb_full)
                v_log_probs = F.log_softmax(v_logits, dim=1)

                v_soft = get_gaussian_labels(
                    vb_y, V, sigma, device,
                    v_rows=SMOOTH_V_ROWS, v_cols=SMOOTH_V_COLS,
                    row_map=BEAM_ROW_MAP, col_map=BEAM_COL_MAP
                )
                v_kl  = kl_criterion(v_log_probs, v_soft)
                v_ce  = ce_criterion(v_logits, vb_y)

                v_soft_fixed = get_gaussian_labels(
                    vb_y, V, SIGMA_END, device,
                    v_rows=SMOOTH_V_ROWS, v_cols=SMOOTH_V_COLS,
                    row_map=BEAM_ROW_MAP, col_map=BEAM_COL_MAP
                )
                v_kl_fixed = kl_criterion(v_log_probs, v_soft_fixed)

            batch_val_loss = v_kl.item() + CE_LOSS_WEIGHT * v_ce.item()
            val_loss_sum  += batch_val_loss * B_vb

            batch_val_loss_fixed = v_kl_fixed.item() + CE_LOSS_WEIGHT * v_ce.item()
            val_loss_fixed_sum  += batch_val_loss_fixed * B_vb

            val_correct   += (torch.argmax(v_logits, dim=1) == vb_y).sum().item()
            val_n         += B_vb

        avg_val_loss       = val_loss_sum / val_n
        avg_val_loss_fixed = val_loss_fixed_sum / val_n
        val_acc            = val_correct / val_n * 100.0

    history['loss'].append(avg_total)
    history['kl'].append(avg_kl)
    history['ce'].append(avg_ce)
    history['supcon'].append(avg_supcon)
    history['ewc'].append(0.0)
    history['lr'].append(lr_now)
    history['sigma'].append(sigma)
    history['val_loss'].append(avg_val_loss)
    history['val_loss_fixed'].append(avg_val_loss_fixed)
    history['val_acc'].append(val_acc)
    history['aging_pct'].append(0)
    history['phase'].append(1)

    scheduler_p1.step(avg_val_loss_fixed)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_loss    = avg_val_loss
        best_weights = copy.deepcopy(model.state_dict())
        no_improve   = 0
        flag = f"* best ({val_acc:.1f}%)"
    else:
        no_improve += 1
        flag = f" wait {no_improve}/{PATIENCE}"

    epoch_dur = time.time() - epoch_start
    p1_epoch_count = epoch
    if epoch % 10 == 0 or epoch == 1 or no_improve >= PATIENCE:
        print(f"{epoch:>6} {sigma:>5.2f} {avg_total:>9.5f} {avg_kl:>9.5f} "
              f"{avg_ce:>9.5f} {avg_supcon:>9.5f} {val_acc:>8.2f}% "
              f"{lr_now:>9.1e} {flag} ({epoch_dur:.2f}s)")

    if no_improve >= PATIENCE:
        print(f"\nPhase 1 early stop at epoch {epoch}. Best val_acc = {best_val_acc:.2f}%")
        break

model.load_state_dict(best_weights)
best_epoch_p1 = history['val_acc'].index(best_val_acc) + 1
print(f"Phase 1 weights restored from epoch {best_epoch_p1} (val_acc = {best_val_acc:.2f}%)")

# Store Phase 1 best for comparison
p1_best_val_acc = best_val_acc
p1_best_loss    = best_loss


# ==============================================================
# CELL 11c: COMPUTE FISHER INFORMATION [D-3]
# ==============================================================
print(f"\nComputing Fisher information for EWC ({EWC_FISHER_BATCHES} batches)...")
fisher_dict, fisher_samples = compute_fisher(
    model, train_loader, budget_tensor, n_batches=EWC_FISHER_BATCHES, dev=device
)
p1_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

fisher_norms = {n: f.norm().item() for n, f in fisher_dict.items()}
fmax_name = max(fisher_norms, key=fisher_norms.get)
fmin_name = min(fisher_norms, key=fisher_norms.get)
print(f"  Fisher computed over {fisher_samples:,} samples")
print(f"  Fisher norm range: {fisher_norms[fmin_name]:.6f} ({fmin_name[:30]}...) "
      f"→ {fisher_norms[fmax_name]:.6f} ({fmax_name[:30]}...)")


# ==============================================================
# CELL 11d: PHASE 2 TRAINING — Curriculum Aging + EWC + Relabelling
# [D-2] Warmup + aging ramp
# [D-3] EWC regularisation
# [D-4] SupCon disabled on aged batches
# [D-5] Relabelling with aged-channel oracle
# [PHYSICS FIX] Aging uses constant thermal noise (not SNR-sampled)
# ==============================================================
print(f"\n{'=' * 90}")
print(f"  PHASE 2: Curriculum aging + Relabelling + EWC ({PHASE2_MAX_EPOCHS} ep)")
print(f"  [D-2] Warmup: {PHASE2_WARMUP_EPOCHS} ep static, then aging "
      f"{PHASE2_AGING_PROB_START}→{PHASE2_AGING_PROB_END}")
print(f"  [D-3] EWC λ={EWC_LAMBDA}")
print(f"  [D-4] SupCon DISABLED on aged batches")
print(f"  [D-5] Aged batches relabelled with aged-channel oracle")
print(f"  [3GPP] Noise: link-budget-aware (σ_noise = {NOISE_STD:.4e})")
print(f"  Curriculum: {CURRICULUM_SPEED_START}→{AGING_MAX_SPEED} m/s")
print(f"{'=' * 90}")

optimizer_p2 = optim.AdamW(
    model.parameters(), lr=LR * PHASE2_LR_FACTOR, weight_decay=WEIGHT_DECAY
)
scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_p2, mode='max', patience=12, factor=0.5
)

# P2 early stopping tracks val_acc (same as P1)
best_val_acc_p2 = best_val_acc   # start from P1 best
best_weights_p2 = copy.deepcopy(best_weights)
no_improve_p2   = 0

print(f"{'Ep':>5} {'σ':>5} {'Loss':>9} {'KL':>9} {'CE':>9} {'SC':>9} {'EWC':>9} "
      f"{'Val Acc':>8} {'lr':>9} {'age%':>5} {'v_mx':>4} status")
print("-" * 120)

model.train()
for ep2 in range(1, PHASE2_MAX_EPOCHS + 1):
    global_ep = p1_epoch_count + ep2
    t0 = time.time()
    frac  = (global_ep - 1) / max(MAX_EPOCHS - 1, 1)
    sigma = max(SIGMA_END, SIGMA_START - (SIGMA_START - SIGMA_END) * frac)

    # [D-2] Warmup: first PHASE2_WARMUP_EPOCHS are static-only
    in_warmup = (ep2 <= PHASE2_WARMUP_EPOCHS)

    if in_warmup:
        current_aging_prob = 0.0
        curr_max_speed = 0.0
    else:
        post_warmup_ep = ep2 - PHASE2_WARMUP_EPOCHS
        post_warmup_total = PHASE2_MAX_EPOCHS - PHASE2_WARMUP_EPOCHS
        ramp_frac = min(post_warmup_ep / max(post_warmup_total * PHASE2_AGING_RAMP_FRAC, 1), 1.0)
        current_aging_prob = (PHASE2_AGING_PROB_START
                              + (PHASE2_AGING_PROB_END - PHASE2_AGING_PROB_START) * ramp_frac)
        p2_frac = (ep2 - 1) / max(PHASE2_MAX_EPOCHS - 1, 1)
        speed_ramp = min(p2_frac / CURRICULUM_RAMP_FRAC, 1.0)
        curr_max_speed = (CURRICULUM_SPEED_START
                          + (AGING_MAX_SPEED - CURRICULUM_SPEED_START) * speed_ramp)

    epoch_loss_total  = 0.0
    epoch_loss_kl     = 0.0
    epoch_loss_ce     = 0.0
    epoch_loss_supcon = 0.0
    epoch_loss_ewc    = 0.0
    n_batches = 0
    n_aged_batches = 0

    model.train()
    for xb, yb, idx_b in train_loader:
        B = xb.shape[0]
        budget_idx = torch.randint(len(BUDGET_LIST), (B,), device=device)
        M_vals     = budget_tensor[budget_idx]
        rand_ranks = torch.rand(B, V, device=device).argsort(dim=1)
        mask       = (rand_ranks < M_vals.unsqueeze(1)).float()

        do_age = (not in_warmup) and (torch.rand(1).item() < current_aging_prob)

        if do_age:
            # ── Aged batch [D-4][D-5] ──
            n_aged_batches += 1
            speeds = torch.rand(B, device=device) * (curr_max_speed - 0.5) + 0.5
            fds    = speeds * FC_HZ / _C_LIGHT

            dt_probe = M_vals.float() / 2 * T_SLOT_FIXED_S
            dt_data  = M_vals.float() * T_SLOT_FIXED_S
            rho_probe = clarke_rho_tensor(fds, dt_probe, RHO_FLOOR)
            rho_data  = clarke_rho_tensor(fds, dt_data,  RHO_FLOOR)

            hb = h_train_gpu[idx_b]

            # Probing through aged channel
            h_probed = age_channel_ar1_batched(hb, rho_probe, device)
            rx_aged  = torch.matmul(h_probed, PROBE_MATRIX.T)

            # [3GPP] Apply exact mathematically scaled SNR
            pl_batch_db = PL_TRAIN_DB[idx_b]
            pl_batch_lin = 10 ** (pl_batch_db / 10)
            p_rx_batch = P_TX_W / pl_batch_lin

            p_rx_safe_batch = p_rx_batch * (NOISE_MULTIPLIER ** 2)
            rx_amp_batch = torch.sqrt(p_rx_safe_batch).unsqueeze(1)

            rx_power_aged = (rx_aged.abs() ** 2).mean(dim=1, keepdim=True)
            rx_scaled = rx_amp_batch * (rx_aged / torch.sqrt(rx_power_aged + 1e-30))

            meas_noise_aged = SAFE_NOISE_FLOOR * (
                    torch.randn_like(rx_scaled.real) + 1j * torch.randn_like(rx_scaled.real)
            )
            rx_noisy = rx_scaled + meas_noise_aged
            lm_aged  = 10 * torch.log10(rx_noisy.abs() + 1e-9)
            # [v8] Instance normalisation: subtract per-sample mean, divide by global std
            lm_sample_means = lm_aged.mean(dim=1, keepdim=True)
            masked_aged = ((lm_aged - lm_sample_means) / X_STD) * mask

            # [D-5] Relabel with aged-channel oracle
            h_data     = age_channel_ar1_batched(hb, rho_data, device)
            aged_bp    = torch.abs(torch.matmul(h_data, CODEBOOK.conj().T)) ** 2
            yb_aged    = aged_bp.argmax(1)
            yb_use     = yb_aged

            xb_full = torch.cat([masked_aged, mask, (M_vals.float() / V).unsqueeze(1)], dim=1)

            optimizer_p2.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # [D-4] No projection — skip SupCon for aged batch
                logits = model(xb_full)
                log_probs    = F.log_softmax(logits, dim=1)
                soft_targets = get_gaussian_labels(
                    yb_use, V, sigma, device,
                    v_rows=SMOOTH_V_ROWS, v_cols=SMOOTH_V_COLS,
                    row_map=BEAM_ROW_MAP, col_map=BEAM_COL_MAP
                )
                loss_kl     = kl_criterion(log_probs, soft_targets)
                loss_ce     = ce_criterion(logits, yb_use)
                loss_task   = loss_kl + CE_LOSS_WEIGHT * loss_ce
                loss_supcon_val = torch.tensor(0.0)

        else:
            # ── Static batch — full loss including SupCon ──
            masked_xb = xb * mask
            yb_use    = yb
            xb_full   = torch.cat([masked_xb, mask, (M_vals.float() / V).unsqueeze(1)], dim=1)

            optimizer_p2.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, proj_feats = model.forward_with_proj(xb_full)
                log_probs    = F.log_softmax(logits, dim=1)
                soft_targets = get_gaussian_labels(
                    yb_use, V, sigma, device,
                    v_rows=SMOOTH_V_ROWS, v_cols=SMOOTH_V_COLS,
                    row_map=BEAM_ROW_MAP, col_map=BEAM_COL_MAP
                )
                loss_kl     = kl_criterion(log_probs, soft_targets)
                loss_ce     = ce_criterion(logits, yb_use)
                loss_supcon_val = supcon_loss(proj_feats.float(), yb_use, temperature=SUPCON_TEMP)
                loss_task   = loss_kl + CE_LOSS_WEIGHT * loss_ce + SUPCON_WEIGHT * loss_supcon_val

        # [D-3] EWC penalty (always applied, outside autocast)
        loss_ewc = ewc_penalty(model, fisher_dict, p1_params, EWC_LAMBDA)
        loss = loss_task + loss_ewc

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_p2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer_p2)
        scaler.update()

        epoch_loss_total  += loss.item()
        epoch_loss_kl     += loss_kl.item()
        epoch_loss_ce     += loss_ce.item()
        epoch_loss_supcon += loss_supcon_val.item()
        epoch_loss_ewc    += loss_ewc.item()
        n_batches += 1

    avg_total  = epoch_loss_total  / n_batches
    avg_kl     = epoch_loss_kl     / n_batches
    avg_ce     = epoch_loss_ce     / n_batches
    avg_supcon = epoch_loss_supcon / n_batches
    avg_ewc    = epoch_loss_ewc    / n_batches
    aging_pct  = n_aged_batches / n_batches * 100
    lr_now     = optimizer_p2.param_groups[0]['lr']

    # ── Phase 2 validation (same dual-metric as P1) ──
    model.eval()
    with torch.no_grad():
        val_loss_sum       = 0.0
        val_loss_fixed_sum = 0.0
        val_correct  = 0
        val_n        = 0

        for vi in range(0, len(X_val_gpu), BATCH):
            vb_x_raw = X_val_gpu[vi:vi + BATCH]
            vb_y     = y_val_gpu[vi:vi + BATCH]
            B_vb     = vb_x_raw.shape[0]

            vb_budget_idx = torch.randint(len(BUDGET_LIST), (B_vb,), device=device)
            vb_M_vals     = budget_tensor[vb_budget_idx]
            vb_rand_ranks = torch.rand(B_vb, V, device=device).argsort(dim=1)
            vb_mask       = (vb_rand_ranks < vb_M_vals.unsqueeze(1)).float()
            vb_masked     = vb_x_raw * vb_mask
            vb_budget_sc  = (vb_M_vals.float() / V).unsqueeze(1)
            vb_full       = torch.cat([vb_masked, vb_mask, vb_budget_sc], dim=1)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                v_logits    = model(vb_full)
                v_log_probs = F.log_softmax(v_logits, dim=1)

                v_soft = get_gaussian_labels(
                    vb_y, V, sigma, device,
                    v_rows=SMOOTH_V_ROWS, v_cols=SMOOTH_V_COLS,
                    row_map=BEAM_ROW_MAP, col_map=BEAM_COL_MAP
                )
                v_kl  = kl_criterion(v_log_probs, v_soft)
                v_ce  = ce_criterion(v_logits, vb_y)

                v_soft_fixed = get_gaussian_labels(
                    vb_y, V, SIGMA_END, device,
                    v_rows=SMOOTH_V_ROWS, v_cols=SMOOTH_V_COLS,
                    row_map=BEAM_ROW_MAP, col_map=BEAM_COL_MAP
                )
                v_kl_fixed = kl_criterion(v_log_probs, v_soft_fixed)

            val_loss_sum       += (v_kl.item() + CE_LOSS_WEIGHT * v_ce.item()) * B_vb
            val_loss_fixed_sum += (v_kl_fixed.item() + CE_LOSS_WEIGHT * v_ce.item()) * B_vb
            val_correct        += (torch.argmax(v_logits, dim=1) == vb_y).sum().item()
            val_n              += B_vb

        avg_val_loss       = val_loss_sum / val_n
        avg_val_loss_fixed = val_loss_fixed_sum / val_n
        val_acc            = val_correct / val_n * 100.0

    history['loss'].append(avg_total)
    history['kl'].append(avg_kl)
    history['ce'].append(avg_ce)
    history['supcon'].append(avg_supcon)
    history['ewc'].append(avg_ewc)
    history['lr'].append(lr_now)
    history['sigma'].append(sigma)
    history['val_loss'].append(avg_val_loss)
    history['val_loss_fixed'].append(avg_val_loss_fixed)
    history['val_acc'].append(val_acc)
    history['aging_pct'].append(aging_pct)
    history['phase'].append(2)

    scheduler_p2.step(val_acc)

    if val_acc > best_val_acc_p2:
        best_val_acc_p2 = val_acc
        best_weights_p2 = copy.deepcopy(model.state_dict())
        no_improve_p2   = 0
        flag = f"* best ({val_acc:.1f}%)"
    else:
        no_improve_p2 += 1
        flag = f" wait {no_improve_p2}/{PHASE2_PATIENCE}"

    warmup_tag = " [WARMUP]" if in_warmup else ""
    epoch_dur = time.time() - t0
    if ep2 % 5 == 0 or ep2 <= PHASE2_WARMUP_EPOCHS or no_improve_p2 >= PHASE2_PATIENCE:
        print(f"{global_ep:>5} {sigma:>5.2f} {avg_total:>9.5f} {avg_kl:>9.5f} "
              f"{avg_ce:>9.5f} {avg_supcon:>9.5f} {avg_ewc:>9.5f} "
              f"{val_acc:>7.2f}% {lr_now:>9.1e} {aging_pct:>4.0f}% "
              f"{curr_max_speed:>4.1f} {flag}{warmup_tag} ({epoch_dur:.2f}s)")

    if no_improve_p2 >= PHASE2_PATIENCE:
        print(f"\nPhase 2 early stop at epoch {global_ep}. Best val_acc = {best_val_acc_p2:.2f}%")
        break

model.load_state_dict(best_weights_p2)
print(f"\nPhase 2 complete.")
print(f"  P1 best val_acc: {p1_best_val_acc:.2f}%  (epoch {best_epoch_p1})")
print(f"  P2 best val_acc: {best_val_acc_p2:.2f}%")
p2_improved = best_val_acc_p2 > p1_best_val_acc
print(f"  Phase 2 {'IMPROVED' if p2_improved else 'did not improve'} over Phase 1")
if p2_improved:
    print(f"  Improvement: {best_val_acc_p2 - p1_best_val_acc:+.2f} pp")

train_end = time.time()


# ==============================================================
# CELL 11e: TRAINING CURVES
# ==============================================================
fig, axes = plt.subplots(2, 4, figsize=(28, 8))

# Row 0: losses
axes[0, 0].plot(history['loss'],   label='Total',  color='navy')
axes[0, 0].plot(history['kl'],     label='KL',     color='steelblue', alpha=0.7)
axes[0, 0].plot(history['ce'],     label='CE',     color='coral',     alpha=0.7)
axes[0, 0].plot(history['supcon'], label='SupCon', color='darkorange', alpha=0.85)
axes[0, 0].axvline(p1_epoch_count, color='green', ls=':', label='P2 start')
if PHASE2_WARMUP_EPOCHS > 0:
    axes[0, 0].axvline(p1_epoch_count + PHASE2_WARMUP_EPOCHS, color='cyan', ls=':', alpha=0.5,
                        label='Warmup end')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss (all components)')
axes[0, 0].legend(fontsize=7); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history['supcon'], color='darkorange')
axes[0, 1].axvline(p1_epoch_count, color='green', ls=':')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('SupCon Loss')
axes[0, 1].set_title('Supervised Contrastive Loss'); axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(history['ewc'], color='brown')
axes[0, 2].axvline(p1_epoch_count, color='green', ls=':')
axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('EWC Penalty')
axes[0, 2].set_title('EWC Penalty'); axes[0, 2].grid(True, alpha=0.3)

axes[0, 3].plot(history['val_acc'], label='Val Acc', color='crimson', alpha=0.9)
axes[0, 3].axvline(p1_epoch_count, color='green', ls=':', label='P2 start')
axes[0, 3].set_xlabel('Epoch'); axes[0, 3].set_ylabel('Top-1 Accuracy (%)')
axes[0, 3].set_title('Validation Accuracy (early-stop)')
axes[0, 3].legend(fontsize=7); axes[0, 3].grid(True, alpha=0.3)

# Row 1: LR, sigma, aging %, val_loss_fixed
axes[1, 0].plot(history['lr'], color='green')
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('LR')
axes[1, 0].set_title('Learning Rate'); axes[1, 0].set_yscale('log'); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history['sigma'], color='purple')
axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Sigma')
axes[1, 1].set_title('Label Smoothing Sigma'); axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(history['aging_pct'], color='crimson')
axes[1, 2].axvline(p1_epoch_count, color='green', ls=':')
if PHASE2_WARMUP_EPOCHS > 0:
    axes[1, 2].axvline(p1_epoch_count + PHASE2_WARMUP_EPOCHS, color='cyan', ls=':', alpha=0.5)
axes[1, 2].set_xlabel('Epoch'); axes[1, 2].set_ylabel('Aging %')
axes[1, 2].set_title('Phase 2 Aging Fraction'); axes[1, 2].grid(True, alpha=0.3)

axes[1, 3].plot(history['val_loss_fixed'], label=f'Val KL+CE (σ={SIGMA_END})', color='teal')
axes[1, 3].axvline(p1_epoch_count, color='green', ls=':')
axes[1, 3].set_xlabel('Epoch'); axes[1, 3].set_ylabel('Val Loss (fixed σ)')
axes[1, 3].set_title(f'Val Loss fixed σ={SIGMA_END} (LR sched)')
axes[1, 3].legend(fontsize=7); axes[1, 3].grid(True, alpha=0.3)

plt.suptitle(f'Training — {N_ROWS}×{N_COLS} RIS {FC_HZ/1e9:.0f} GHz {BITS_LABEL} '
             f'V={V} + SupCon + EWC + Curriculum', y=1.02)
plt.tight_layout()
plt.savefig(get_save_path('Training.png')); plt.close()
print("Saved: Training.png")


# ==============================================================
# CELL 12: STATIC EVALUATION
# [FIX-3] Average over N_EVAL_MASKS random probe masks per budget
# ==============================================================
def make_masked_input(X_full, M, v_size, dev, seed=0):
    """[zero-masked probes (V), binary mask (V), budget scalar (1)]"""
    torch.manual_seed(seed)
    S    = X_full.shape[0]
    x    = X_full.clone().to(dev)
    perm = torch.randperm(v_size, device=dev)
    mask = torch.zeros(v_size, device=dev)
    mask[perm[:M]] = 1.0
    mask_batch = mask.unsqueeze(0).expand(S, -1)
    x = x * mask_batch
    budget = torch.full((S, 1), M / v_size, dtype=torch.float32, device=dev)
    return torch.cat([x, mask_batch, budget], dim=1)


model.eval()
topk_header = ' '.join([f'Top-{k}' for k in TOP_K_LIST])
print(f"\n{'=' * 90}")
print(f"  STATIC EVALUATION  [FIX-3] Averaging over {N_EVAL_MASKS} masks")
print(f"  {'M':>6} {topk_header} {'SE Ratio':>10} {'SE (dB)':>8} "
      f"{'Rate pred':>10} {'Rate oracle':>11}")
print(f"{'=' * 90}")

results_static = {}
for M in sorted(set(BUDGET_LIST)):
    topk_accum = {k: [] for k in TOP_K_LIST}
    se_ratio_accum  = []
    rate_pred_accum = []
    rate_oracle_val = None

    for mask_seed in range(N_EVAL_MASKS):
        x_eval    = make_masked_input(X_test, M, V, device, seed=mask_seed)
        all_logits = []; all_preds = []
        with torch.no_grad():
            for i in range(0, len(x_eval), EVAL_BATCH_SIZE):
                batch_logits = model(x_eval[i:i+EVAL_BATCH_SIZE].to(device))
                all_logits.append(batch_logits)
                all_preds.append(torch.argmax(batch_logits, dim=1))
        logits_cat = torch.cat(all_logits)
        preds      = torch.cat(all_preds)
        yg         = y_test.to(device)

        for k in TOP_K_LIST:
            topk_accum[k].append(top_k_accuracy(logits_cat, yg, k) * 100)

        p_pred     = beam_power(h_test, preds, CODEBOOK)
        p_oracle   = beam_power(h_test, yg, CODEBOOK)
        se_ratio_accum.append(
            (p_pred / (p_oracle + 1e-30)).clamp(0, 1).mean().item() * 100
        )
        rate_pred_accum.append(achievable_rate(p_pred, SNR_LINEAR).mean().item())
        if rate_oracle_val is None:
            rate_oracle_val = achievable_rate(p_oracle, SNR_LINEAR).mean().item()

    topk_accs  = {k: float(np.mean(topk_accum[k])) for k in TOP_K_LIST}
    topk_stds  = {k: float(np.std(topk_accum[k]))  for k in TOP_K_LIST}
    se_ratio   = float(np.mean(se_ratio_accum))
    se_dB      = 10 * np.log10(se_ratio / 100 + 1e-9)
    rate_pred  = float(np.mean(rate_pred_accum))
    rate_oracle = rate_oracle_val

    results_static[M] = dict(topk=topk_accs, topk_std=topk_stds,
                             se_ratio=se_ratio, se_dB=se_dB,
                             rate_pred=rate_pred, rate_oracle=rate_oracle)

    topk_str = ' '.join([f'{topk_accs[k]:>5.2f}%' for k in TOP_K_LIST])
    std_str  = ' '.join([f'±{topk_stds[k]:.2f}' for k in TOP_K_LIST])
    print(f"  M={M:<4} {topk_str} {se_ratio:>9.2f}% {se_dB:>7.2f} dB "
          f"{rate_pred:>8.2f} b/s/Hz {rate_oracle:>8.2f} b/s/Hz")
    print(f"         {std_str}   (std over {N_EVAL_MASKS} masks)")
print(f"{'=' * 90}")

# Static accuracy plot
ms = sorted(results_static.keys())
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
colors_k = ['navy', 'steelblue', 'skyblue']
for i, k in enumerate(TOP_K_LIST):
    means = [results_static[m]['topk'][k] for m in ms]
    stds  = [results_static[m]['topk_std'][k] for m in ms]
    axes[0].errorbar(ms, means, yerr=stds,
                     fmt='o-', lw=2, capsize=3,
                     color=colors_k[i % len(colors_k)], label=f'Top-{k}')
axes[0].set_xlabel('Budget M'); axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title(f'Top-K Accuracy vs Budget (±std, {N_EVAL_MASKS} masks)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(ms, [results_static[m]['se_ratio'] for m in ms], 's-r', lw=2)
axes[1].set_xlabel('Budget M'); axes[1].set_ylabel('SE Ratio (%)')
axes[1].set_title('Spectral Efficiency Ratio vs Budget'); axes[1].grid(True, alpha=0.3)
axes[2].plot(ms, [results_static[m]['rate_pred']   for m in ms], 'o-', lw=2, color='green', label='Predicted')
axes[2].plot(ms, [results_static[m]['rate_oracle'] for m in ms], '--', lw=2, color='gray',  label='Oracle')
axes[2].set_xlabel('Budget M'); axes[2].set_ylabel('Rate (bits/s/Hz)')
axes[2].set_title('Achievable Rate vs Budget'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
plt.suptitle(f'{BITS_LABEL} — {N_ROWS}×{N_COLS} RIS {FC_HZ/1e9:.0f} GHz + SupCon (Static)', y=1.02)
plt.tight_layout()
plt.savefig(get_save_path('Static_Evaluation.png')); plt.close()
print("Saved: Static_Evaluation.png")


# ==============================================================
# CELL 13: DOPPLER EVALUATION [D-7]
# [3GPP] Uses link-budget-aware noise for consistency
# ==============================================================
print(f"\n{'=' * 105}")
print(f"  DOPPLER EVALUATION  AR(1) ρ=max(|J₀|,{RHO_FLOOR})  T_slot={T_SLOT_FIXED_S*1e6:.1f}µs")
print(f"  [3GPP] Link-budget-aware noise (σ_noise={NOISE_STD:.4e})")
print(f"{'=' * 105}")


def eval_doppler(h_base, y_lab, M, fd, speed_mps, mdl, cb, pm, vs, x_std,
                 snr_lin, dev, pl_test_db, seed=RANDOM_SEED):
    """Evaluate beam selection under Doppler with 3GPP link budget noise."""
    S = h_base.shape[0]
    hd = h_base.to(dev)
    pd_ = pm.to(dev)
    rp_ = clarke_rho(fd, M / 2 * T_SLOT_FIXED_S)
    rd_ = clarke_rho(fd, M * T_SLOT_FIXED_S)

    # [D-6] Physics-correct overhead
    tc_s = get_coherence_time_s(speed_mps)
    probing_time_s = M * T_SLOT_FIXED_S
    oh_physics = probing_time_s / tc_s if tc_s < float('inf') else 0.0
    exceeds_tc = probing_time_s > tc_s

    # Age channel for probing phase
    torch.manual_seed(seed)
    hp = age_channel_ar1(hd, rp_, dev)
    rx = torch.matmul(hp, pd_.T)

    # [3GPP] Apply exact mathematically scaled SNR
    pl_lin = 10 ** (pl_test_db.to(dev) / 10)
    p_rx = P_TX_W / pl_lin

    p_rx_safe_eval = p_rx * (NOISE_MULTIPLIER ** 2)
    rx_amp_eval = torch.sqrt(p_rx_safe_eval).unsqueeze(1)

    rx_pwr = (rx.abs() ** 2).mean(dim=1, keepdim=True)
    rx_scaled = rx_amp_eval * (rx / torch.sqrt(rx_pwr + 1e-30))

    meas_noise_eval = SAFE_NOISE_FLOOR * (
            torch.randn_like(rx_scaled.real) + 1j * torch.randn_like(rx_scaled.real)
    )
    rxn = rx_scaled + meas_noise_eval
    lm  = 10 * torch.log10(rxn.abs() + 1e-9)
    # [v8] Instance normalisation
    lm_means = lm.mean(dim=1, keepdim=True)
    xf  = ((lm - lm_means) / x_std).float()

    # Apply mask
    torch.manual_seed(seed + 1000)
    perm = torch.randperm(vs, device=dev)
    am = torch.zeros(vs, device=dev)
    am[perm[:M]] = 1
    mb = am.unsqueeze(0).expand(S, -1)
    xi = torch.cat([xf * mb, mb, torch.full((S, 1), M / vs, device=dev)], 1)

    # Predict
    lg = []; pr = []
    mdl.eval()
    with torch.no_grad():
        for i in range(0, S, EVAL_BATCH_SIZE):
            bl = mdl(xi[i:i+EVAL_BATCH_SIZE])
            lg.append(bl); pr.append(bl.argmax(1))
    lc = torch.cat(lg); preds = torch.cat(pr); yg = y_lab.to(dev)
    tk = {k: top_k_accuracy(lc, yg, k) * 100 for k in TOP_K_LIST}

    # Age channel for data phase
    torch.manual_seed(seed + 2000)
    hdata = age_channel_ar1(hd, rd_, dev)

    pp      = (hdata.to(cb.device) * cb[preds.to(cb.device)].conj()).sum(1).abs().pow(2)
    ob      = cb[yg.to(cb.device)]
    po_orig = (hd.to(cb.device) * ob.conj()).sum(1).abs().pow(2)
    po_aged = (hdata.to(cb.device) * ob.conj()).sum(1).abs().pow(2)
    aged_bp = torch.abs(torch.matmul(hdata, cb.conj().T)) ** 2
    po_adapt = aged_bp.max(1)[0]

    aged_oracle_labels = aged_bp.argmax(1)
    tk_aged = {k: top_k_accuracy(lc, aged_oracle_labels, k) * 100 for k in TOP_K_LIST}

    se = (pp / (po_orig + 1e-30)).clamp(0, 1).mean().item() * 100
    eff_mult = max(1.0 - oh_physics, 0.0)

    return dict(
        topk=tk, topk_aged=tk_aged,
        se_ratio=se, se_dB=10 * np.log10(se / 100 + 1e-9),
        raw_rate=achievable_rate(pp, snr_lin).mean().item(),
        oracle_rate=achievable_rate(po_orig, snr_lin).mean().item(),
        oracle_aged_rate=achievable_rate(po_aged, snr_lin).mean().item(),
        oracle_adapt_rate=achievable_rate(po_adapt, snr_lin).mean().item(),
        eff_rate=eff_mult * achievable_rate(pp, snr_lin).mean().item(),
        eff_oracle=eff_mult * achievable_rate(po_orig, snr_lin).mean().item(),
        rho_probe=rp_, rho_data=rd_,
        oh_physics=oh_physics, exceeds_tc=exceeds_tc,
        probing_time_us=probing_time_s * 1e6,
        coherence_time_us=tc_s * 1e6 if tc_s < float('inf') else float('inf'),
    )


all_doppler_results = {}
for nm, spd in SPEED_SCENARIOS.items():
    fd = spd * FC_HZ / _C_LIGHT if spd > 1e-6 else 0.0
    tc = '∞' if fd < 1e-6 else f'{0.423/fd*1e3:.2f}'
    max_M_feas = get_max_feasible_M(spd, T_SLOT_FIXED_S)
    print(f"\n  {nm:>12s}: v={spd:.1f}  f_D={fd:.0f} Hz  T_c={tc} ms  max_M={max_M_feas}")
    print(f"  {'M':>5} " + ' '.join([f'Top-{k}' for k in TOP_K_LIST]) +
          f"  {'T1_age':>6} {'SE%':>5} {'R_raw':>6} {'R_eff':>6} {'R_adapt':>7} "
          f"{'ρ_d':>5} {'Oh%':>5} {'Feas':>4}")
    print(f"  {'-' * 110}")
    sr = {}
    for M in sorted(set(BUDGET_LIST)):
        r = eval_doppler(h_test, y_test, M, fd, spd, model, CODEBOOK, PROBE_MATRIX,
                         V, X_STD.item(), SNR_LINEAR, device,
                         PL_TEST_DB)
        sr[M] = r
        feas_tag = "✓" if not r['exceeds_tc'] else "✗"
        ts = ' '.join([f'{r["topk"][k]:>5.1f}%' for k in TOP_K_LIST])
        print(f"  M={M:<4}{ts} {r['topk_aged'][1]:>5.1f}% {r['se_ratio']:>5.1f}% "
              f"{r['raw_rate']:>6.2f} {r['eff_rate']:>6.2f} {r['oracle_adapt_rate']:>7.2f} "
              f"{r['rho_data']:>.3f} {r['oh_physics']*100:>5.1f} {feas_tag}")
    all_doppler_results[nm] = sr
print(f"\n{'=' * 105}")


# ==============================================================
# CELL 14: DOPPLER PLOTS
# ==============================================================
print("\nGenerating Doppler plots...")
sc_col = {'static': 'gray', 'pedestrian': 'steelblue', 'cyclist': 'darkorange', 'vehicular': 'crimson'}
sc_mk  = {'static': 's', 'pedestrian': 'o', 'cyclist': '^', 'vehicular': 'D'}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
for nm in SPEED_SCENARIOS:
    ax.plot(ms, [all_doppler_results[nm][m]['topk'][1] for m in ms],
            marker=sc_mk[nm], lw=2, color=sc_col[nm], label=nm)
ax.set_xlabel('M'); ax.set_ylabel('Top-1 (%)'); ax.set_title('Top-1 (vs t=0 oracle)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for nm in SPEED_SCENARIOS:
    ax.plot(ms, [all_doppler_results[nm][m]['topk_aged'][1] for m in ms],
            marker=sc_mk[nm], lw=2, color=sc_col[nm], label=nm)
ax.set_xlabel('M'); ax.set_ylabel('Top-1 (%)'); ax.set_title('Top-1 (vs aged oracle)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
for nm in SPEED_SCENARIOS:
    ax.plot(ms, [all_doppler_results[nm][m]['se_ratio'] for m in ms],
            marker=sc_mk[nm], lw=2, color=sc_col[nm], label=nm)
ax.set_xlabel('M'); ax.set_ylabel('SE (%)'); ax.set_title('SE Ratio')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for nm in SPEED_SCENARIOS:
    ax.plot(ms, [all_doppler_results[nm][m]['eff_rate'] for m in ms],
            marker=sc_mk[nm], lw=2, color=sc_col[nm], label=nm)
    for i, m in enumerate(ms):
        if all_doppler_results[nm][m]['exceeds_tc']:
            ax.plot(m, all_doppler_results[nm][m]['eff_rate'], 'x', color='black',
                    markersize=12, markeredgewidth=2)
ax.plot(ms, [all_doppler_results['static'][m]['eff_oracle'] for m in ms],
        '--k', lw=2, alpha=0.5, label='Oracle')
ax.set_xlabel('M'); ax.set_ylabel('R_eff')
ax.set_title('Effective Rate (✗=exceeds T_c)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
s1 = {m: all_doppler_results['static'][m]['topk'][1] for m in ms}
for nm in SPEED_SCENARIOS:
    if nm == 'static': continue
    ax.plot(ms, [s1[m] - all_doppler_results[nm][m]['topk'][1] for m in ms],
            marker=sc_mk[nm], lw=2, color=sc_col[nm], label=nm)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('M'); ax.set_ylabel('Drop (pp)'); ax.set_title('Degradation')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 2]
for nm in SPEED_SCENARIOS:
    raw   = [all_doppler_results[nm][m]['raw_rate'] for m in ms]
    adapt = [all_doppler_results[nm][m]['oracle_adapt_rate'] for m in ms]
    ax.plot(ms, raw, marker=sc_mk[nm], lw=2, color=sc_col[nm], label=f'{nm}')
    ax.plot(ms, adapt, marker=sc_mk[nm], ls='--', lw=1, color=sc_col[nm], alpha=0.4)
ax.set_xlabel('M'); ax.set_ylabel('Rate')
ax.set_title('Raw vs Adapted Oracle'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle(f'Doppler — {BITS_LABEL} H={HIDDEN}×{N_RESBLOCKS} 3GPP UMi-SC + EWC', y=1.02)
plt.tight_layout()
plt.savefig(get_save_path('Doppler_Comparison.png'), dpi=150, bbox_inches='tight'); plt.close()
print("Saved: Doppler_Comparison.png")

# Autocorrelation decay plot
if _HAS_SCIPY:
    from scipy.special import j0 as bj0
    fig, ax = plt.subplots(figsize=(8, 5))
    tau = np.linspace(0, 2, 500)
    for nm, spd in SPEED_SCENARIOS.items():
        if spd < 1e-6:
            ax.axhline(1, color=sc_col[nm], ls='--', label=nm, alpha=0.6)
            continue
        fd_ = spd * FC_HZ / _C_LIGHT
        rho_a = np.array([max(abs(bj0(2*np.pi*fd_*t*1e-3)), RHO_FLOOR) for t in tau])
        ax.plot(tau, rho_a, color=sc_col[nm], lw=2, label=f'{nm} (f_D={fd_:.0f}Hz)')
    for Mm in [32, 128, 256, 453]:
        tm = Mm * T_SLOT_FIXED_S * 1e3
        ax.axvline(tm, color='gray', ls=':', alpha=0.4)
        ax.text(tm + 0.01, RHO_FLOOR + 0.02, f'M={Mm}', fontsize=7, color='gray')
    ax.axhline(RHO_FLOOR, color='red', ls='--', alpha=0.3, label=f'floor={RHO_FLOOR}')
    ax.set_xlabel('τ (ms)'); ax.set_ylabel('ρ_eff'); ax.set_title('Effective ρ')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, 2); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(get_save_path('Autocorrelation_Decay.png'), dpi=150); plt.close()
    print("Saved: Autocorrelation_Decay.png")

# Heatmaps
scm = [n for n in SPEED_SCENARIOS if n != 'static']
fig, axes = plt.subplots(1, 3, figsize=(20, 4))

rh = np.array([[all_doppler_results[n][m]['rho_data'] for m in ms] for n in scm])
im = axes[0].imshow(rh, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
axes[0].set_xticks(range(len(ms))); axes[0].set_xticklabels(ms)
axes[0].set_yticks(range(len(scm))); axes[0].set_yticklabels(scm)
for i in range(len(scm)):
    for j in range(len(ms)):
        axes[0].text(j, i, f'{rh[i,j]:.3f}', ha='center', va='center', fontsize=9)
axes[0].set_xlabel('M'); axes[0].set_title('ρ_data'); plt.colorbar(im, ax=axes[0])

eh = np.array([[all_doppler_results[n][m]['eff_rate'] for m in ms] for n in scm])
im = axes[1].imshow(eh, aspect='auto', cmap='YlOrRd_r')
axes[1].set_xticks(range(len(ms))); axes[1].set_xticklabels(ms)
axes[1].set_yticks(range(len(scm))); axes[1].set_yticklabels(scm)
for i in range(len(scm)):
    for j in range(len(ms)):
        axes[1].text(j, i, f'{eh[i,j]:.1f}', ha='center', va='center', fontsize=9)
axes[1].set_xlabel('M'); axes[1].set_title('R_eff'); plt.colorbar(im, ax=axes[1])

oh_h = np.array([[all_doppler_results[n][m]['oh_physics']*100 for m in ms] for n in scm])
im = axes[2].imshow(oh_h, aspect='auto', cmap='Reds', vmin=0)
axes[2].set_xticks(range(len(ms))); axes[2].set_xticklabels(ms)
axes[2].set_yticks(range(len(scm))); axes[2].set_yticklabels(scm)
for i in range(len(scm)):
    for j in range(len(ms)):
        txt = f'{oh_h[i,j]:.0f}%'
        if oh_h[i,j] > 100: txt += '\n⚠'
        axes[2].text(j, i, txt, ha='center', va='center', fontsize=8)
axes[2].set_xlabel('M'); axes[2].set_title('Overhead % (>100%=infeasible)')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig(get_save_path('Heatmaps.png'), dpi=150); plt.close()
print("Saved: Heatmaps.png")

# Oracle analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for nm in SPEED_SCENARIOS:
    if nm == 'static': continue
    oa = [all_doppler_results[nm][m]['oracle_aged_rate'] for m in ms]
    oo = [all_doppler_results[nm][m]['oracle_rate'] for m in ms]
    ax.plot(ms, [100*(1-a/o) if o > 0 else 0 for a, o in zip(oa, oo)],
            marker=sc_mk[nm], lw=2, color=sc_col[nm], label=nm)
ax.set_xlabel('M'); ax.set_ylabel('Oracle Loss (%)')
ax.set_title('t=0 Oracle Beam Aging Loss'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
for nm in SPEED_SCENARIOS:
    if nm == 'static': continue
    ax.plot(ms, [all_doppler_results[nm][m]['oracle_adapt_rate']
                 - all_doppler_results[nm][m]['oracle_aged_rate'] for m in ms],
            marker=sc_mk[nm], lw=2, color=sc_col[nm], label=nm)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('M'); ax.set_ylabel('Gap')
ax.set_title('Adapted−Stale Oracle'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(get_save_path('Oracle_Analysis.png'), dpi=150); plt.close()
print("Saved: Oracle_Analysis.png")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
xp = np.arange(len(ms)); w = 0.2
for i, nm in enumerate(SPEED_SCENARIOS):
    ax.bar(xp + i*w, [all_doppler_results[nm][m]['topk'][1] for m in ms],
           w, label=nm, color=sc_col[nm], alpha=0.85)
ax.set_xticks(xp + 1.5*w); ax.set_xticklabels([f'M={m}' for m in ms])
ax.set_ylabel('Top-1 (%)'); ax.set_title('Accuracy by Speed')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(get_save_path('Accuracy_BarChart.png'), dpi=150); plt.close()
print("Saved: Accuracy_BarChart.png")


# ==============================================================
# CELL 15: SNR ROBUSTNESS TEST
# [v8] Uses SAFE_NOISE_FLOOR (same as training) + instance norm.
# Each SNR level sets a uniform per-sample signal amplitude so
# that mean SNR = target. Instance norm erases absolute power,
# so the test verifies the network learned spatial patterns,
# not absolute signal level.
# ==============================================================
print(f"\n{'=' * 90}")
M_test = min(128, V)
print(f"  SNR ROBUSTNESS TEST (M={M_test}, static)")
print(f"  [v8] Instance-normalised features, SAFE_NOISE_FLOOR={SAFE_NOISE_FLOOR}")
print(f"{'=' * 90}")
test_snr_list = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]

h_test_dev   = h_test.to(device)
probe_dev    = PROBE_MATRIX.to(device)
received_clean = torch.matmul(h_test_dev, probe_dev.T)

for t_snr in test_snr_list:
    # Target SNR in the SAFE tensor space (noise variance = SAFE_NOISE_FLOOR² * 2)
    target_snr_lin = 10 ** (t_snr / 10.0)
    noise_power_safe = SAFE_NOISE_FLOOR ** 2 * 2  # consistent with training
    target_signal_power = target_snr_lin * noise_power_safe

    # Scale clean probes to achieve target SNR
    clean_mean_power = received_clean.abs().pow(2).mean().item()
    scale_factor = float(np.sqrt(target_signal_power / (clean_mean_power + 1e-30)))

    torch.manual_seed(RANDOM_SEED)
    rx_scaled = received_clean * scale_factor
    meas_noise_test = SAFE_NOISE_FLOOR * (
        torch.randn_like(rx_scaled.real) + 1j * torch.randn_like(rx_scaled.real)
    )
    received_noisy_test = rx_scaled + meas_noise_test
    log_mag_test = 10 * torch.log10(received_noisy_test.abs() + 1e-9)

    # [v8] Instance normalisation — same as training
    sample_means_test = log_mag_test.mean(dim=1, keepdim=True)
    X_test_snr = (log_mag_test - sample_means_test) / X_STD.to(device)

    x_eval_snr = make_masked_input(X_test_snr, M_test, V, device)
    preds_snr = []

    with torch.no_grad():
        for i in range(0, len(x_eval_snr), EVAL_BATCH_SIZE):
            logits_snr = model(x_eval_snr[i:i + EVAL_BATCH_SIZE])
            preds_snr.append(torch.argmax(logits_snr, dim=1))

    preds_snr = torch.cat(preds_snr)
    acc_snr = (preds_snr == y_test.to(device)).float().mean().item() * 100

    print(f"  Test SNR = {t_snr:>5.1f} dB  |  "
          f"Top-1 Accuracy (M={M_test}) = {acc_snr:>5.2f}%")
print(f"{'=' * 90}")


# ==============================================================
# CELL 16: ONLINE INFERENCE DEMO
# [PHYSICS FIX] Uses constant NOISE_STD.
# ==============================================================
def online_predict(h_sample, M, mdl, probe_mat, v_size, x_std, dev,
                   fd=0.0, pl_db=None):
    """Predict best beam from M magnitude probes with 3GPP link budget + instance norm."""
    mdl.eval()
    with torch.no_grad():
        h = h_sample.to(dev).unsqueeze(0)

        # Age channel if mobile
        rho_p = clarke_rho(fd, M / 2 * T_SLOT_FIXED_S)
        h_probed = age_channel_ar1(h, rho_p, dev)

        y_all_clean = probe_mat.to(dev) @ h_probed.squeeze(0)

        # Apply link budget in safe tensor space
        if pl_db is not None:
            pl_lin = 10 ** (pl_db / 10)
            p_rx = P_TX_W / pl_lin
            p_rx_safe = p_rx * (NOISE_MULTIPLIER ** 2)
            rx_amp = float(np.sqrt(p_rx_safe))
            clean_pwr = (y_all_clean.abs() ** 2).mean().item()
            y_all_clean = rx_amp * y_all_clean / float(np.sqrt(clean_pwr + 1e-30))

        meas_noise = SAFE_NOISE_FLOOR * (
            torch.randn_like(y_all_clean.real) + 1j * torch.randn_like(y_all_clean.real)
        )
        y_all_noisy = y_all_clean + meas_noise

        perm = torch.randperm(v_size, device=dev)
        active_idx = perm[:M]
        probes = torch.zeros(v_size, dtype=torch.float32, device=dev)

        log_mag = 10 * torch.log10(y_all_noisy[active_idx].abs() + 1e-9)
        # [v8] Instance normalisation: subtract mean of active probes
        sample_mean = log_mag.mean()
        probes[active_idx] = (log_mag - sample_mean) / x_std

        mask_vec = torch.zeros(v_size, dtype=torch.float32, device=dev)
        mask_vec[active_idx] = 1.0
        budget = torch.tensor([M / v_size], dtype=torch.float32, device=dev)
        x_in = torch.cat([probes, mask_vec, budget]).unsqueeze(0)

        return torch.argmax(mdl(x_in), dim=1).item()


print(f"\nOnline Inference Demo — RIS_BITS={RIS_BITS} ({BITS_LABEL})")
print(f"  [3GPP] Using link-budget-aware noise (σ_noise = {NOISE_STD:.4e})")
print(f"{'=' * 75}")
n_demo = min(ONLINE_DEMO_SAMPLES, len(idx_test))
for nm, spd in SPEED_SCENARIOS.items():
    fd = spd * FC_HZ / _C_LIGHT if spd > 1e-6 else 0.0
    print(f"\n  {nm} (v={spd:.1f} m/s, f_D={fd:.0f} Hz)")
    for Md in sorted(set(BUDGET_LIST)):
        correct = 0
        for i in range(n_demo):
            pl_sample = PL_TEST_DB[i].item()
            pred = online_predict(
                cascade_channel[idx_test[i]], Md, model, PROBE_MATRIX,
                V, X_STD.item(), device, fd,
                pl_db=pl_sample
            )
            if pred == y_test[i].item():
                correct += 1
        acc = correct / n_demo * 100
        print(f"    M={Md:<4}  {acc:>5.1f}%  ({correct}/{n_demo})")
print(f"{'=' * 75}")


# ==============================================================
# CELL 17: SAVE TRAINED MODEL
# ==============================================================
save_path = get_save_path(f'ris_model_v8_3gpp_{RIS_BITS}bit.pth')
checkpoint = {
    'model_state_dict'      : model.state_dict(),
    'INPUT_DIM'             : INPUT_DIM,
    'HIDDEN'                : HIDDEN,
    'N_RESBLOCKS'           : N_RESBLOCKS,
    'DROPOUT_RATE'          : DROPOUT_RATE,
    'LEAKY_SLOPE'           : LEAKY_SLOPE,
    'PROJ_DIM'              : PROJ_DIM,
    'SUPCON_WEIGHT'         : SUPCON_WEIGHT,
    'SUPCON_TEMP'           : SUPCON_TEMP,
    'V'                     : V,
    'V_ORIGINAL'            : V_ORIGINAL,
    'input_format'          : '2V+1: [probes(V), mask(V), budget(1)]',
    'N_ROWS'                : N_ROWS,
    'N_COLS'                : N_COLS,
    'V_ROWS'                : V_ROWS,
    'V_COLS'                : V_COLS,
    'N'                     : N,
    'FC_HZ'                 : FC_HZ,
    'WAVELENGTH'            : WAVELENGTH,
    'ELEM_SPACING'          : ELEM_SPACING,
    # 3GPP link budget parameters
    'P_TX_DBM'              : P_TX_DBM,
    'NOISE_STD'             : NOISE_STD,
    'N0_W'                  : N0_W,
    'K_BOLTZMANN'           : K_BOLTZMANN,
    'T_SYS_K'               : T_SYS_K,
    'BW_HZ'                 : BW_HZ,
    'NOISE_FIGURE_DB'       : NOISE_FIGURE_DB,
    'D_BS_RIS_MIN'          : D_BS_RIS_MIN,
    'D_BS_RIS_MAX'          : D_BS_RIS_MAX,
    'D_RIS_UE_MIN'          : D_RIS_UE_MIN,
    'D_RIS_UE_MAX'          : D_RIS_UE_MAX,
    'RIS_ELEM_AMPLITUDE'    : RIS_ELEM_AMPLITUDE,
    'channel_model'         : 'ETSI GR RIS 003 + 3GPP TR 38.901 SF',
    'normalisation'         : 'instance (per-sample mean subtracted)',
    'TEST_SNR_DB'           : TEST_SNR_DB,
    'SNR_LINEAR'            : SNR_LINEAR,
    'X_std'                 : X_STD.item(),
    'SAFE_NOISE_FLOOR'      : SAFE_NOISE_FLOOR,
    'NOISE_MULTIPLIER'      : NOISE_MULTIPLIER,
    'PROBE_MATRIX'          : PROBE_MATRIX.cpu(),
    'CODEBOOK'              : CODEBOOK.cpu(),
    'CODEBOOK_UNIQUE_IDX'   : CODEBOOK_UNIQUE_IDX.cpu(),
    'RIS_BITS'              : RIS_BITS,
    'PHASE_SET'             : PHASE_SET.cpu() if PHASE_SET is not None else None,
    'BITS_LABEL'            : BITS_LABEL,
    'BUDGET_LIST'           : BUDGET_LIST,
    'SMOOTH_V_ROWS'         : SMOOTH_V_ROWS,
    'SMOOTH_V_COLS'         : SMOOTH_V_COLS,
    'BEAM_ROW_MAP'          : BEAM_ROW_MAP.cpu() if BEAM_ROW_MAP is not None else None,
    'BEAM_COL_MAP'          : BEAM_COL_MAP.cpu() if BEAM_COL_MAP is not None else None,
    # Doppler parameters
    'T_SLOT_FIXED_S'        : T_SLOT_FIXED_S,
    'RHO_FLOOR'             : RHO_FLOOR,
    'SPEED_SCENARIOS'       : SPEED_SCENARIOS,
    'EWC_LAMBDA'            : EWC_LAMBDA,
    # Training metadata
    'best_epoch_p1'         : best_epoch_p1,
    'p1_best_val_acc'       : p1_best_val_acc,
    'p2_best_val_acc'       : best_val_acc_p2,
    'VAL_FRAC'              : VAL_FRAC,
    'history'               : history,
    'results_static'        : results_static,
    'results_doppler'       : all_doppler_results,
    'MAX_EPOCHS'            : MAX_EPOCHS,
    'PHASE1_MAX_EPOCHS'     : PHASE1_MAX_EPOCHS,
    'PHASE2_MAX_EPOCHS'     : PHASE2_MAX_EPOCHS,
    'BATCH'                 : BATCH,
    'LR'                    : LR,
    'WEIGHT_DECAY'          : WEIGHT_DECAY,
    'SIGMA_START'           : SIGMA_START,
    'SIGMA_END'             : SIGMA_END,
    'CE_LOSS_WEIGHT'        : CE_LOSS_WEIGHT,
    'NUM_SAMPLES'           : NUM_SAMPLES,
    'RANDOM_SEED'           : RANDOM_SEED,
    'N_EVAL_MASKS'          : N_EVAL_MASKS,
    'TOP_K_LIST'            : TOP_K_LIST,
    'saved_at'              : datetime.datetime.now().isoformat(),
}
torch.save(checkpoint, save_path)
sz_kb = os.path.getsize(save_path) / 1024
print(f"\nSaved -> {save_path} ({sz_kb:.1f} KB)")


# ==============================================================
# CELL 18: FULL EXPERIMENT REPORT
# ==============================================================
bud = sorted(set(BUDGET_LIST))
W = 120
lines = [
    '=' * W,
    f"  {'RIS BEAM SELECTION — FULL EXPERIMENT REPORT (v8 — ETSI/3GPP)'.center(W-2)}",
    f"  {'ETSI GR RIS 003 + Instance Norm + Doppler'.center(W-2)}",
    '=' * W,
    f"  Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"  Array      : {N_ROWS} × {N_COLS} = {N} elements   V = {V} beams",
    f"  Frequency  : {FC_HZ/1e9:.1f} GHz",
    f"  RIS_BITS   : {RIS_BITS} ({BITS_LABEL})",
    f"  Model      : HIDDEN={HIDDEN} × {N_RESBLOCKS} ResBlocks  params={n_params:,}",
    f"  Batch      : {BATCH}  AMP=bfloat16",
    f"  Loss       : KL + {CE_LOSS_WEIGHT}*CE + {SUPCON_WEIGHT}*SupCon(T={SUPCON_TEMP})",
    f"  Phase 1    : {p1_epoch_count} ep (best ep {best_epoch_p1}, val_acc={p1_best_val_acc:.2f}%)",
    f"  Phase 2    : curriculum+EWC+relabel+no_aged_supcon (val_acc={best_val_acc_p2:.2f}%)",
    f"  --- ETSI GR RIS 003 Link Budget ---",
    f"  P_TX       : {P_TX_DBM:.1f} dBm",
    f"  BS→RIS     : d₁ ∈ [{D_BS_RIS_MIN:.0f},{D_BS_RIS_MAX:.0f}] m",
    f"  RIS→UE     : d₂ ∈ [{D_RIS_UE_MIN:.0f},{D_RIS_UE_MAX:.0f}] m",
    f"  PL formula : (4π d₁ d₂ / (N·dx·dy·A))²   ETSI GR RIS 003",
    f"  PL(mean)   : {_PL_RIS_MEAN_DB:.1f} dB  σ_SF={SIGMA_SF_COMBINED_DB} dB",
    f"  Noise      : {N0_DBM:.1f} dBm  (T={T_SYS_K:.0f}K, BW={BW_HZ/1e6:.0f}MHz, NF={NOISE_FIGURE_DB}dB)",
    f"  Mean SNR   ≈ {_SNR_MEAN_DB:.1f} dB",
    f"  --- Training ---",
    f"  EWC        : λ={EWC_LAMBDA}",
    f"  Aging      : AR(1) ρ=max(|J₀|,{RHO_FLOOR}) T_slot={T_SLOT_FIXED_S*1e6:.1f}µs",
    f"  P2 warmup  : {PHASE2_WARMUP_EPOCHS} ep, aging ramp: "
    f"{PHASE2_AGING_PROB_START}→{PHASE2_AGING_PROB_END}",
    f"  Split      : train={1-TEST_FRAC-VAL_FRAC:.0%} / val={VAL_FRAC:.0%} / test={TEST_FRAC:.0%}",
    f"  Eval masks : N_EVAL_MASKS={N_EVAL_MASKS}",
    f"  Output     : {os.path.abspath(OUTPUT_DIR)}",
    '',
]

# Scenario info
for nm, spd in SPEED_SCENARIOS.items():
    fd = spd * FC_HZ / _C_LIGHT if spd > 1e-6 else 0.0
    tc = '∞' if fd < 1e-6 else f'{0.423/fd*1e3:.3f}ms'
    max_M_f = get_max_feasible_M(spd, T_SLOT_FIXED_S)
    lines.append(f"  {nm:>12s}: v={spd:>5.1f} f_D={fd:>7.1f}Hz T_c={tc:>10s} max_M={max_M_f}")

# Static results
lines += ['', '-' * W, '', '  ── STATIC EVALUATION ──']
lines.append(f"  {'M':>6} " + ' '.join([f'Top-{k:>1}' for k in TOP_K_LIST]) +
             f"  {'SE Ratio':>9} {'SE dB':>7} {'Rate pred':>10} {'Rate oracle':>11}")
lines.append(f"  {'-' * 80}")
for M in bud:
    r = results_static[M]
    topk_str = ' '.join([f'{r["topk"][k]:>5.1f}%' for k in TOP_K_LIST])
    lines.append(
        f"  M={M:<4} {topk_str}  {r['se_ratio']:>8.2f}%  {r['se_dB']:>6.2f} "
        f"  {r['rate_pred']:>8.2f} b/s  {r['rate_oracle']:>8.2f} b/s"
    )

# Doppler results
lines += ['', '-' * W]
for nm in SPEED_SCENARIOS:
    lines.append(f"\n  ── {nm.upper()} ──")
    lines.append(f"  {'M':>5} " + ' '.join([f'Top-{k}' for k in TOP_K_LIST]) +
                 f" {'T1_age':>6} {'SE%':>5} {'R_raw':>6} {'R_eff':>6} "
                 f"{'R_adapt':>7} {'ρ_d':>5} {'Oh%':>5} {'Feas':>4}")
    lines.append(f"  {'-' * 90}")
    for M in bud:
        r = all_doppler_results[nm][M]
        feas = "✓" if not r['exceeds_tc'] else "✗"
        ts = ' '.join([f'{r["topk"][k]:>5.1f}%' for k in TOP_K_LIST])
        lines.append(
            f"  M={M:<4}{ts} {r['topk_aged'][1]:>5.1f}% {r['se_ratio']:>5.1f}% "
            f"{r['raw_rate']:>6.2f} {r['eff_rate']:>6.2f} {r['oracle_adapt_rate']:>7.2f} "
            f"{r['rho_data']:>.3f} {r['oh_physics']*100:>5.1f} {feas}"
        )

# Key findings
lines += ['', '=' * W, f"  {'KEY FINDINGS'.center(W-2)}", '=' * W, '']
lines.append("  WHY ACCURACY DROPS WITH MORE PROBES AT HIGH SPEED:")
lines.append("  ─────────────────────────────────────────────────")
lines.append(f"  Each probe costs {T_SLOT_FIXED_S*1e6:.1f}µs. "
             f"Total probing time = M × {T_SLOT_FIXED_S*1e6:.1f}µs.")
lines.append(f"  Channel correlation at data phase: ρ = max(|J₀(2πf_D·M·T_slot)|, {RHO_FLOOR})")
lines.append(f"  When ρ < 0.5, probes describe a STALE channel → beam selection fails.")
lines.append(f"")
lines.append(f"  TIMING CONSTRAINT: Probing time must not exceed coherence time.")
lines.append(f"  When M·T_slot > T_c, probes themselves span multiple coherence intervals,")
lines.append(f"  making measurements internally inconsistent. Such M values are marked ✗.")
lines.append('')

for M in [64, 128, 256]:
    if M not in bud: continue
    s1_ = all_doppler_results['static'][M]['topk'][1]
    for nm in ['pedestrian', 'cyclist', 'vehicular']:
        a1_ = all_doppler_results[nm][M]['topk'][1]
        rd  = all_doppler_results[nm][M]['rho_data']
        t1a = all_doppler_results[nm][M]['topk_aged'][1]
        ad  = all_doppler_results[nm][M]['oracle_adapt_rate']
        feas = "✓" if not all_doppler_results[nm][M]['exceeds_tc'] else "✗"
        lines.append(
            f"  M={M} {nm:>12s}: {s1_:.1f}%→{a1_:.1f}% (Δ={s1_-a1_:+.1f}pp) "
            f"ρ={rd:.3f} T1_aged={t1a:.1f}% R_adapt={ad:.1f} {feas}"
        )

lines.append('')
lines.append("  OPTIMAL M* per scenario (maximising R_eff, feasible only):")
for nm in SPEED_SCENARIOS:
    feasible_ms = [m for m in bud if not all_doppler_results[nm][m]['exceeds_tc']]
    if feasible_ms:
        bm = max(feasible_ms, key=lambda m: all_doppler_results[nm][m]['eff_rate'])
        be_ = all_doppler_results[nm][bm]['eff_rate']
        rd  = all_doppler_results[nm][bm]['rho_data']
        lines.append(f"    {nm:>12s}: M*={bm} R_eff={be_:.2f} ρ={rd:.3f}")
    else:
        lines.append(f"    {nm:>12s}: NO feasible M")

lines += ['', '=' * W, f"  Model: {save_path}", '=' * W]
report = '\n'.join(lines)
print(report)
with open(get_save_path('experiment_report.txt'), 'w') as f:
    f.write(report)
print(f"\nSaved: {get_save_path('experiment_report.txt')}")


# ==============================================================
# CELL 19: TIMING REPORT
# ==============================================================
script_end_time = time.time()
total_duration  = script_end_time - script_start_time
hours, rem   = divmod(total_duration, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\n{'=' * 60}")
print(f"  TOTAL EXECUTION COMPLETED")
print(f"{'=' * 60}")
print(f"  Total Time   : {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
print(f"  Training Time: {train_end - train_start:.2f} s")
print(f"  Model saved  : {save_path}")
print(f"  Output       : {os.path.abspath(OUTPUT_DIR)}")
print(f"{'=' * 60}")
print("Done.")