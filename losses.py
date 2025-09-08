# models/losses.py
# ------------------------------------------------------------
# Paper-exact losses (no SSIM/LPIPS/aux). 그대로 구현.
# - Stage-1: L_stage1 = || I - D(E(I)) ||_2^2
# - Stage-2:
#     L_simple = || eps_t - eps_theta(x_t, ~x, t) ||_2^2
#     L_diff   = L_simple + || P̂_low - P_high ||_1
#     L_fre    = || Â_low - A_high ||_2^2
#     L_con    = || Î_low - I_high ||_2^2 + || F̂_low - F_high ||_2^2
#     L_total  = L_diff + λ1 * L_con + λ2 * L_fre
# ------------------------------------------------------------
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


# ---------------- Stage-1 ----------------
def loss_stage1_autoencoder(I: torch.Tensor, I_rec: torch.Tensor) -> torch.Tensor:
    """L_stage1 = || I - D(E(I)) ||_2^2"""
    return F.mse_loss(I_rec, I, reduction="mean")


# ---------------- Stage-2: 구성요소 ----------------
def loss_diffusion_simple(eps_pred: torch.Tensor, eps_gt: torch.Tensor) -> torch.Tensor:
    """L_simple = || eps_t - eps_theta(x_t, ~x, t) ||_2^2"""
    return F.mse_loss(eps_pred, eps_gt, reduction="mean")


def loss_phase_l1(P_hat_low: torch.Tensor, P_high: torch.Tensor) -> torch.Tensor:
    """|| P̂_low - P_high ||_1  (각도 텐서(rad) 기준)"""
    return F.l1_loss(P_hat_low, P_high, reduction="mean")


def loss_freq_amp_l2(A_hat_low: torch.Tensor, A_high: torch.Tensor) -> torch.Tensor:
    """L_fre = || Â_low - A_high ||_2^2"""
    return F.mse_loss(A_hat_low, A_high, reduction="mean")


def loss_content_l2(
    I_hat_low: torch.Tensor, I_high: torch.Tensor,
    F_hat_low: torch.Tensor, F_high: torch.Tensor
) -> torch.Tensor:
    """L_con = || Î_low - I_high ||_2^2 + || F̂_low - F_high ||_2^2"""
    l_img = F.mse_loss(I_hat_low, I_high, reduction="mean")
    l_lat = F.mse_loss(F_hat_low, F_high, reduction="mean")
    return l_img + l_lat


# ---------------- Stage-2: 집계 ----------------
def loss_diff(
    eps_pred: torch.Tensor, eps_gt: torch.Tensor,
    P_hat_low: torch.Tensor, P_high: torch.Tensor
) -> torch.Tensor:
    """L_diff = L_simple + || P̂_low - P_high ||_1"""
    Lsimple = loss_diffusion_simple(eps_pred, eps_gt)
    Lphase  = loss_phase_l1(P_hat_low, P_high)
    return Lsimple + Lphase


def loss_stage2_total(
    *,
    eps_pred: torch.Tensor, eps_gt: torch.Tensor,       # diffusion
    P_hat_low: torch.Tensor, P_high: torch.Tensor,      # phase
    A_hat_low: torch.Tensor, A_high: torch.Tensor,      # amplitude
    I_hat_low: torch.Tensor, I_high: torch.Tensor,      # image
    F_hat_low: torch.Tensor, F_high: torch.Tensor,      # latent/feature
    lambda1: float = 1.0,    # weight for L_con
    lambda2: float = 1.0     # weight for L_fre
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    반환:
      L_total, {'L_diff':..., 'L_simple':..., 'L_phase':..., 'L_con':..., 'L_fre':...}
    """
    # 구성요소
    Lsimple = loss_diffusion_simple(eps_pred, eps_gt)
    Lphase  = loss_phase_l1(P_hat_low, P_high)
    Ldiff   = Lsimple + Lphase
    Lcon    = loss_content_l2(I_hat_low, I_high, F_hat_low, F_high)
    Lfre    = loss_freq_amp_l2(A_hat_low, A_high)

    # 총합
    L_total = Ldiff + lambda1 * Lcon + lambda2 * Lfre

    logs = {
        "L_total": L_total.detach(),
        "L_diff":  Ldiff.detach(),
        "L_simple": Lsimple.detach(),
        "L_phase": Lphase.detach(),
        "L_con":   Lcon.detach(),
        "L_fre":   Lfre.detach(),
    }
    return L_total, logs
