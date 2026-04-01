# -*- coding: utf-8 -*-
"""
cdl_38901_ris.py

Standalone 3GPP TR 38.901 CDL-style RIS channel generator (narrowband geometric form).

Design goals:
- Generate BS->RIS (H1) and RIS->UE (H2) separately using CDL-style clusters/rays.
- Cascade into RIS element-domain vector h in C^{S x N}.
- Keep unit-order amplitudes (no link budget/noise applied here).

This module intentionally does NOT include:
- ETSI path loss
- thermal noise
- transmit power scaling
Those are handled by the caller (e.g., probing/link-budget stage).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch


# ---------------------------------------------------------------------
# 3GPP TR 38.901 CDL-C table (cluster means) commonly used in practice.
# Columns: [delay_norm, power_dB, AOD_deg, AOA_deg, ZOD_deg, ZOA_deg]
# ---------------------------------------------------------------------
CDL_C_TABLE = np.array([
    [0.0000, -4.4,  -46.6, -101.0,  97.2,  87.6],
    [0.2099, -1.2,  -22.8,  120.0,  98.6,  72.1],
    [0.2219, -3.5,  -22.8,  120.0,  98.6,  72.1],
    [0.2329, -5.2,  -22.8,  120.0,  98.6,  72.1],
    [0.2176, -2.5,  -40.7, -127.5, 100.6,  70.1],
    [0.6366,  0.0,    0.3,  170.4,  99.2,  75.3],
    [0.6448, -2.2,    0.3,  170.4,  99.2,  75.3],
    [0.6560, -3.9,    0.3,  170.4,  99.2,  75.3],
    [0.6584, -7.4,   73.1,   55.4, 105.2,  67.4],
    [0.7935, -7.1,  -64.5,   66.5,  95.3,  63.8],
    [0.8213,-10.7,   80.2,  -48.1, 106.1,  71.4],
    [0.9336,-11.1,  -97.1,   46.9,  93.5,  60.5],
    [1.2285, -5.1,  -55.3,   68.1, 103.7,  90.6],
    [1.3083, -6.8,  -64.3,  -68.7, 104.2,  60.1],
    [2.1704, -8.7,  -78.5,   81.5,  93.0,  61.0],
    [2.7105,-13.2,  102.7,   30.7, 104.2, 100.7],
    [4.2589,-13.9,   99.2,  -16.4,  94.9,  62.3],
    [4.6003,-13.9,   88.8,    3.8,  93.1,  66.7],
    [5.4902,-15.8, -101.9,  -13.7,  92.2,  52.9],
    [5.6077,-17.1,   92.2,    9.7, 106.7,  61.8],
    [6.3065,-16.0,   93.3,    5.6,  93.0,  51.9],
    [6.6374,-15.7,  106.6,    0.7,  92.9,  61.7],
    [7.0427,-21.6,  119.5,  -21.9, 105.2,  58.0],
    [8.6523,-22.8, -123.8,   33.6, 107.8,  57.0],
], dtype=np.float32)

# CDL ray offset set (20 rays, symmetric)
CDL_RAY_OFFSETS_20 = np.array([
    -0.8844,  0.8844, -0.6797,  0.6797, -0.5246,  0.5246, -0.3715,  0.3715,
    -0.2230,  0.2230, -0.0766,  0.0766, -0.0433,  0.0433, -0.1331,  0.1331,
    -0.2835,  0.2835, -0.4365,  0.4365
], dtype=np.float32)


@dataclass
class CDLConfig:
    fc_hz: float
    n_rows: int
    n_cols: int
    c_asd_deg: float = 2.0
    c_asa_deg: float = 15.0
    c_zsd_deg: float = 3.0
    c_zsa_deg: float = 7.0
    n_rays: int = 20
    chunk_size: int = 256
    random_global_rotation: bool = False
    shared_cluster_geometry: bool = False
    eps_norm: float = 1e-12


class CDL38901RIS:
    """
    3GPP CDL-style RIS geometric generator (narrowband).
    """

    def __init__(self, cfg: CDLConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.c_light = 3e8
        self.wavelength = self.c_light / cfg.fc_hz
        self.k_wave = 2 * np.pi / self.wavelength
        self.elem_spacing = self.wavelength / 2.0

        self.N = cfg.n_rows * cfg.n_cols
        self.C = CDL_C_TABLE.shape[0]
        assert cfg.n_rays == 20, "This module currently uses standard 20-ray CDL offsets."

        # Element coordinates (y,z) for UPA in yz-plane
        row_idx = torch.arange(cfg.n_rows, device=device, dtype=torch.float32).repeat_interleave(cfg.n_cols)
        col_idx = torch.arange(cfg.n_cols, device=device, dtype=torch.float32).repeat(cfg.n_rows)
        self.elem_y = ((col_idx - (cfg.n_cols - 1) / 2.0) * self.elem_spacing).unsqueeze(0)  # (1,N)
        self.elem_z = ((row_idx - (cfg.n_rows - 1) / 2.0) * self.elem_spacing).unsqueeze(0)  # (1,N)

        # Cluster means
        self.aod_c = torch.tensor(CDL_C_TABLE[:, 2], device=device, dtype=torch.float32).view(1, self.C, 1)
        self.aoa_c = torch.tensor(CDL_C_TABLE[:, 3], device=device, dtype=torch.float32).view(1, self.C, 1)
        self.zod_c = torch.tensor(CDL_C_TABLE[:, 4], device=device, dtype=torch.float32).view(1, self.C, 1)
        self.zoa_c = torch.tensor(CDL_C_TABLE[:, 5], device=device, dtype=torch.float32).view(1, self.C, 1)

        # Power weights
        p_db = torch.tensor(CDL_C_TABLE[:, 1], device=device, dtype=torch.float32)
        p_lin = 10.0 ** (p_db / 10.0)
        self.p_cluster = (p_lin / p_lin.sum()).view(1, self.C, 1, 1)  # (1,C,1,1)
        self.ray_amp = torch.sqrt(self.p_cluster / cfg.n_rays)         # (1,C,1,1)

        # Ray offsets
        self.ray_off = torch.tensor(CDL_RAY_OFFSETS_20, device=device, dtype=torch.float32).view(1, 1, cfg.n_rays)
        self.d_aod = cfg.c_asd_deg * self.ray_off
        self.d_aoa = cfg.c_asa_deg * self.ray_off
        self.d_zod = cfg.c_zsd_deg * self.ray_off
        self.d_zoa = cfg.c_zsa_deg * self.ray_off

    def steering_vector(self, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """
        azimuth, elevation: (Sx,1)
        return: (Sx,N) complex steering vector
        """
        phase = self.k_wave * (
            self.elem_y * torch.sin(azimuth) * torch.cos(elevation)
            + self.elem_z * torch.sin(elevation)
        )
        return torch.exp(1j * phase)

    @staticmethod
    def _wrap_deg(x: torch.Tensor) -> torch.Tensor:
        return (x + 180.0) % 360.0 - 180.0

    def _build_angles(self, B: int, is_link1: bool = True) -> Dict[str, torch.Tensor]:
        """
        Build (B,C,R) angles for one sub-link.
        Link1 uses AOD/ZOD as transmit side and AOA/ZOA as receive side.
        Link2 reuses CDL profile similarly as another independent sub-link realization.
        """
        if is_link1:
            tx_az_c, rx_az_c = self.aod_c, self.aoa_c
            tx_ze_c, rx_ze_c = self.zod_c, self.zoa_c
        else:
            # independent sub-link, same CDL profile semantics
            tx_az_c, rx_az_c = self.aod_c, self.aoa_c
            tx_ze_c, rx_ze_c = self.zod_c, self.zoa_c

        tx_az = tx_az_c.expand(B, self.C, self.cfg.n_rays) + self.d_aod.expand(B, self.C, self.cfg.n_rays)
        rx_az = rx_az_c.expand(B, self.C, self.cfg.n_rays) + self.d_aoa.expand(B, self.C, self.cfg.n_rays)
        tx_ze = tx_ze_c.expand(B, self.C, self.cfg.n_rays) + self.d_zod.expand(B, self.C, self.cfg.n_rays)
        rx_ze = rx_ze_c.expand(B, self.C, self.cfg.n_rays) + self.d_zoa.expand(B, self.C, self.cfg.n_rays)

        if self.cfg.random_global_rotation:
            rot = torch.rand(B, 1, 1, device=self.device) * 360.0 - 180.0
            tx_az = self._wrap_deg(tx_az + rot)
            rx_az = self._wrap_deg(rx_az + rot)

        # deg -> rad ; zenith -> elevation
        tx_az_rad = tx_az * (np.pi / 180.0)
        rx_az_rad = rx_az * (np.pi / 180.0)
        tx_el_rad = (np.pi / 2.0) - tx_ze * (np.pi / 180.0)
        rx_el_rad = (np.pi / 2.0) - rx_ze * (np.pi / 180.0)

        return dict(
            tx_az_deg=tx_az, rx_az_deg=rx_az, tx_ze_deg=tx_ze, rx_ze_deg=rx_ze,
            tx_az_rad=tx_az_rad, rx_az_rad=rx_az_rad, tx_el_rad=tx_el_rad, rx_el_rad=rx_el_rad
        )

    def _sub_link_from_angles(self, ang: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate a sub-link field from pre-built angle tensors (independent phases only).
        Enables shared cluster geometry across H1 and H2.
        Returns: (B,N) complex64
        """
        B = ang["tx_az_rad"].shape[0]
        sv_tx = self.steering_vector(ang["tx_az_rad"].reshape(-1, 1), ang["tx_el_rad"].reshape(-1, 1)) \
            .reshape(B, self.C, self.cfg.n_rays, self.N)
        sv_rx = self.steering_vector(ang["rx_az_rad"].reshape(-1, 1), ang["rx_el_rad"].reshape(-1, 1)) \
            .reshape(B, self.C, self.cfg.n_rays, self.N)

        phi = torch.rand(B, self.C, self.cfg.n_rays, 1, device=self.device) * 2 * np.pi
        ray_phase = torch.exp(1j * phi)

        h_rays = self.ray_amp * ray_phase * sv_tx * torch.conj(sv_rx)
        h_link = h_rays.sum(dim=2).sum(dim=1)

        p = (h_link.abs() ** 2).mean(dim=1, keepdim=True)
        h_link = h_link / torch.sqrt(p + self.cfg.eps_norm)
        return h_link.to(torch.complex64)

    def _sub_link_field(self, B: int, is_link1: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate one sub-link field over RIS elements: (B,N)
        """
        ang = self._build_angles(B, is_link1=is_link1)

        sv_tx = self.steering_vector(ang["tx_az_rad"].reshape(-1, 1), ang["tx_el_rad"].reshape(-1, 1)) \
            .reshape(B, self.C, self.cfg.n_rays, self.N)
        sv_rx = self.steering_vector(ang["rx_az_rad"].reshape(-1, 1), ang["rx_el_rad"].reshape(-1, 1)) \
            .reshape(B, self.C, self.cfg.n_rays, self.N)

        phi = torch.rand(B, self.C, self.cfg.n_rays, 1, device=self.device) * 2 * np.pi
        ray_phase = torch.exp(1j * phi)

        h_rays = self.ray_amp * ray_phase * sv_tx * torch.conj(sv_rx)  # (B,C,R,N)
        h_link = h_rays.sum(dim=2).sum(dim=1)                          # (B,N)

        # Unit-order normalization per sample
        p = (h_link.abs() ** 2).mean(dim=1, keepdim=True)
        h_link = h_link / torch.sqrt(p + self.cfg.eps_norm)

        return h_link.to(torch.complex64), ang

    def generate(self, num_samples: int, return_debug: bool = False):
        """
        Returns:
            h_cascade: (S,N) complex
            h1: (S,N) complex
            h2: (S,N) complex
            debug (optional): angular tensors for validation
        """
        S = num_samples
        h1_out = torch.empty(S, self.N, device=self.device, dtype=torch.complex64)
        h2_out = torch.empty(S, self.N, device=self.device, dtype=torch.complex64)
        h_out = torch.empty(S, self.N, device=self.device, dtype=torch.complex64)

        dbg = {"link1": {}, "link2": {}} if return_debug else None

        chunk = self.cfg.chunk_size
        for s0 in range(0, S, chunk):
            s1 = min(s0 + chunk, S)
            B = s1 - s0

            if self.cfg.shared_cluster_geometry:
                # Opt-in: shared cluster geometry with independent phases per sub-link.
                # Pure product cascade used when geometry is shared.
                ang_shared = self._build_angles(B, is_link1=True)
                h1 = self._sub_link_from_angles(ang_shared)
                h2 = self._sub_link_from_angles(ang_shared)
                # a1/a2 intentionally share the same reference; the debug loop only reads
                # tensor values and does not modify them in-place.
                a1 = ang_shared
                a2 = ang_shared
                # Pure cascade (appropriate for shared-geometry path)
                h_c = h1 * h2
            else:
                h1, a1 = self._sub_link_field(B, is_link1=True)
                h2, a2 = self._sub_link_field(B, is_link1=False)
                # Geometric cascade with controlled mixing:
                # product term preserves two-hop coupling, sum term preserves coherent structure.
                h_c = 0.7 * (h1 * h2) + 0.3 * (h1 + h2) / np.sqrt(2.0)

            # Re-normalize cascade to unit-order
            p_c = (h_c.abs() ** 2).mean(dim=1, keepdim=True)
            h_c = h_c / torch.sqrt(p_c + self.cfg.eps_norm)

            h1_out[s0:s1] = h1
            h2_out[s0:s1] = h2
            h_out[s0:s1] = h_c

            if return_debug:
                for k, v in a1.items():
                    dbg["link1"].setdefault(k, []).append(v.detach().cpu())
                for k, v in a2.items():
                    dbg["link2"].setdefault(k, []).append(v.detach().cpu())

        if return_debug:
            for lk in ["link1", "link2"]:
                for k in dbg[lk]:
                    dbg[lk][k] = torch.cat(dbg[lk][k], dim=0)  # (S,C,R)
            return h_out, h1_out, h2_out, dbg

        return h_out, h1_out, h2_out