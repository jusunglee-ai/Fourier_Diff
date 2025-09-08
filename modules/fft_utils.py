# modules/fft_modules.py
# ------------------------------------------------------------
# Latent 2D-FFT utilities for LDM-based low-light enhancement
# - Safe casting for FFT (fp16 -> fp32)
# - Amplitude / Phase decomposition
# - Phase <-> (cos, sin) embedding conversion
# - iFFT reconstruction
# - Optional padding helpers (power-of-two / even sizes)
# ------------------------------------------------------------

from __future__ import annotations
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


# -----------------------------
# Casting helpers
# -----------------------------
def _to_float32(x: torch.Tensor) -> torch.Tensor:
    """
    FFT는 fp16에서 수치 불안정/미지원 케이스가 있으므로 fp32로 올려서 처리.
    (torch.fft는 실수 입력이면 복소로 upcast됨)
    """
    if x.dtype not in (torch.float32, torch.float64):
        return x.float()
    return x


def _same_dtype(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """ref의 dtype으로 되돌림(보통 fp16로) — 단, 최종 출력의 안전성을 위해 원하는 경우 유지해도 됨."""
    if x.dtype != ref.dtype:
        return x.to(ref.dtype)
    return x


# -----------------------------
# FFT / IFFT (unitary)
# -----------------------------
def fft2d(z: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    2D-FFT: real z[B,C,H,W] -> complex Z[B,C,H,W]  (unitary 'ortho')
    """
    assert z.dim() == 4, "z must be [B, C, H, W]"
    z32 = _to_float32(z)
    Z = torch.fft.fft2(z32, norm=norm)  # complex64
    return Z


def ifft2d(Z: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    2D-iFFT: complex Z[B,C,H,W] -> real z[B,C,H,W]
    """
    assert Z.is_complex(), "Z must be complex"
    z = torch.fft.ifft2(Z, norm=norm).real
    return z


# -----------------------------
# Amplitude / Phase
# -----------------------------
def amp_phase(z: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    실수 z -> (A, P, Z)
      - A = |Z| >= 0
      - P = angle(Z) in [-pi, pi]
      - Z = complex spectrum (나중에 원하면 사용)
    반환 dtype은 fp32 (안정성)
    """
    Z = fft2d(z)                             # complex
    A = torch.abs(Z).clamp_min(eps)          # [B,C,H,W], fp32
    P = torch.angle(Z)                       # [-pi, pi], fp32
    return A, P, Z


def amp_phase_from_complex(Z: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    복소 스펙트럼 Z -> (A, P)
    """
    assert Z.is_complex(), "Z must be complex"
    A = torch.abs(Z).clamp_min(eps)
    P = torch.angle(Z)
    return A, P


def combine_amp_phase(A: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    (A, P) -> z (iFFT)
    A, P는 fp32 실수 텐서. 출력 z는 fp32.
    """
    Z = A * torch.exp(1j * P)
    z = ifft2d(Z)                            # real
    return z


# -----------------------------
# Phase <-> Embedding (cos/sin)
# -----------------------------
def phase_to_embedding(P: torch.Tensor) -> torch.Tensor:
    """
    위상 P[B,C,H,W] -> 임베딩 E[B,2C,H,W]  (채널 축에 cos/sin concat)
    확산의 회전 불변성 & wrap-around 문제 완화를 위해 cos/sin 사용.
    """
    cosP = torch.cos(P)
    sinP = torch.sin(P)
    E = torch.cat([cosP, sinP], dim=1)
    return E


def embedding_to_phase(E: torch.Tensor) -> torch.Tensor:
    """
    임베딩 E[B,2C,H,W] -> 위상 P[B,C,H,W]
    채널을 반으로 나눠 (cos, sin)로 복원 후 atan2(sin, cos).
    """
    assert E.dim() == 4 and E.size(1) % 2 == 0, "E must be [B, 2C, H, W]"
    C2 = E.size(1) // 2
    cosP, sinP = E[:, :C2], E[:, C2:]
    # 수치적 안정성을 위한 clamp
    cosP = cosP.clamp(min=-1.0, max=1.0)
    sinP = sinP.clamp(min=-1.0, max=1.0)
    P = torch.atan2(sinP, cosP)
    return P


# -----------------------------
# Convenience wrappers
# -----------------------------
def decompose(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    편의 함수: z -> (A, P, E)
      - A: amplitude [B,C,H,W]
      - P: phase     [B,C,H,W]
      - E: phase embedding [B,2C,H,W]
    모두 fp32.
    """
    A, P, _ = amp_phase(z)
    E = phase_to_embedding(P)
    return A, P, E


def reconstruct(A_hat: torch.Tensor, P_embed_hat: torch.Tensor) -> torch.Tensor:
    """
    편의 함수: (Â, Ê) -> z_rec
      - P_embed_hat: [B,2C,H,W]
      - 반환 z_rec: fp32
    """
    P_hat = embedding_to_phase(P_embed_hat)
    z_rec = combine_amp_phase(A_hat, P_hat)
    return z_rec


# -----------------------------
# Optional: padding helpers
# -----------------------------
def pad_to_multiple(x: torch.Tensor, mult: int = 8, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """
    H,W가 mult의 배수가 되도록 패딩. FFT 안정성/속도 & VAE downsample 정합에 유용.
    반환: (x_pad, pad_tuple)  # pad_tuple = (left, right, top, bottom)
    """
    B, C, H, W = x.shape
    pad_h = (mult - H % mult) % mult
    pad_w = (mult - W % mult) % mult
    pad = (0, pad_w, 0, pad_h)  # (left,right,top,bottom) for F.pad with 4D? -> Actually torch.nn.functional.pad expects (left, right, top, bottom)
    x_pad = F.pad(x, pad, mode=mode)
    return x_pad, pad


def unpad(x: torch.Tensor, pad: Tuple[int,int,int,int]) -> torch.Tensor:
    """
    pad_to_multiple로 패딩한 텐서 원복.
    pad=(left,right,top,bottom)
    """
    l, r, t, b = pad
    if (l, r, t, b) == (0, 0, 0, 0):
        return x
    H, W = x.size(-2), x.size(-1)
    return x[..., t:H-b if b>0 else H, l:W-r if r>0 else W]
