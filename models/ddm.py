# models/ddm.py
# -----------------------------------------------------------------------------
# Strict LFDM (Latent-Fourier Diffusion for Low-Light) trainer wrapper
# - Stage-1: AE pretrain (I -> E -> D -> I_rec)
# - Stage-2: Phase diffusion (DDPM full-T) + LBM + iFFT + LFFM + Decoder
# - EMA: included via pipeline model. Use swap_to_ema()/swap_to_raw() for eval.
# - NO external `utils` dependency: minimal helpers included.
# -----------------------------------------------------------------------------
import os
import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from .autoencoder import LightenEncoder, LightenDecoder
from .unet import DiffusionUNet
from .losses import loss_stage1_autoencoder, loss_stage2_total
from modules.Transformer_modules import LBM_Original_ReLU as LBM, LFFM_Original_ReLU as LFFM
from modules.fft_utils import (
    amp_phase, phase_to_embedding, embedding_to_phase, combine_amp_phase,
    pad_to_multiple, unpad
)

# ------------------------ Minimal helpers (no utils) ------------------------
def save_image_tensor(x: torch.Tensor, path: str):
    """x: [-1,1] or [0,1] tensor (B,C,H,W) or (C,H,W). Save first image if batch."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if x.dim() == 4:
        x = x[0]
    # map to [0,1]
    if x.min() < 0:
        x = (x.clamp(-1, 1) + 1) / 2.0
    try:
        from torchvision.utils import save_image
        save_image(x, path)
    except Exception:
        np.save(path + ".npy", x.detach().cpu().numpy())

def save_checkpoint(state: dict, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filename: str, map_location=None):
    return torch.load(filename, map_location=map_location or "cpu")

def make_optimizer(config, params):
    lr = getattr(config.training, "lr", 2e-4)
    wd = getattr(config.training, "weight_decay", 0.0)
    betas = getattr(config.training, "betas", (0.9, 0.999))
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)

# ------------------------ Diffusion schedule helpers ------------------------
def make_beta_schedule(kind: str, beta_start: float, beta_end: float, T: int) -> torch.Tensor:
    kind = kind.lower()
    if kind == "linear":
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    elif kind == "quad":
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, T, dtype=torch.float32)**2
    elif kind == "const":
        betas = torch.full((T,), beta_end, dtype=torch.float32)
    elif kind == "jsd":
        betas = 1.0 / torch.linspace(T, 1, T, dtype=torch.float32)
    elif kind == "sigmoid":
        grid = torch.linspace(-6, 6, T, dtype=torch.float32)
        betas = torch.sigmoid(grid) * (beta_end - beta_start) + beta_start
    else:
        raise ValueError(f"Unknown beta schedule: {kind}")
    return betas

# ------------------------ Config ------------------------
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class DiffusionConfig:
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    num_diffusion_timesteps: int = 1000

    latent_ch: int = 256
    image_size: int = 64

    unet_ch: int = 64
    unet_ch_mult: tuple = (1, 2, 4, 4)
    unet_num_res_blocks: int = 2
    unet_attn_resolutions: tuple = (16,)
    unet_dropout: float = 0.0
    unet_resamp_with_conv: bool = True

    lambda1: float = 1.0
    lambda2: float = 1.0

# ------------------------ EMA ------------------------
class EMAHelper(object):
    def __init__(self, mu: float = 0.9999):
        self.mu = float(mu)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _unwrap(module: nn.Module) -> nn.Module:
        return module.module if isinstance(module, nn.DataParallel) else module

    def register(self, module: nn.Module):
        m = self._unwrap(module)
        self.shadow.clear()
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone()

    def update(self, module: nn.Module):
        m = self._unwrap(module)
        if not self.shadow:
            self.register(m); return
        mu = self.mu
        for name, p in m.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(mu).add_(p.data, alpha=(1.0 - mu))

    def apply_to(self, module: nn.Module):
        m = self._unwrap(module)
        for name, p in m.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])

    def backup(self, module: nn.Module):
        m = self._unwrap(module)
        self.backup_params = {n: p.data.detach().clone() for n,p in m.named_parameters() if p.requires_grad}

    def restore(self, module: nn.Module):
        m = self._unwrap(module)
        if not self.backup_params: return
        for name, p in m.named_parameters():
            if p.requires_grad and name in self.backup_params:
                p.data.copy_(self.backup_params[name])
        self.backup_params.clear()

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}

# ------------------------ Core model ------------------------
class LFDMNetWithEMA(nn.Module):
    # --------------- EMA convenience ---------------
    def start_ema(self):
        if getattr(self, "use_ema", False) and getattr(self, "ema_helper", None) is not None:
            self.ema_helper.register(self)

    def update_ema(self):
        if getattr(self, "use_ema", False) and getattr(self, "ema_helper", None) is not None:
            self.ema_helper.update(self)

    def swap_to_ema(self):
        if getattr(self, "use_ema", False) and getattr(self, "ema_helper", None) is not None:
            self.ema_helper.backup(self)
            self.ema_helper.apply_to(self)

    def swap_to_raw(self):
        if getattr(self, "use_ema", False) and getattr(self, "ema_helper", None) is not None:
            self.ema_helper.restore(self)

    def __init__(self, cfg: DiffusionConfig, *, use_ema: bool = True):
        super().__init__()
        self.cfg = cfg

        # Encoder / Decoder
        self.encoder = LightenEncoder(base_ch=64, latent_ch=cfg.latent_ch)
        self.decoder = LightenDecoder(base_ch=64, latent_ch=cfg.latent_ch, out_act="tanh")

        # LBM / LFFM
        self.lbm  = LBM(channels=cfg.latent_ch)
        self.lffm = LFFM(channels=cfg.latent_ch)

        # Diffusion UNet
        class _Wrap: pass
        M = _Wrap(); D = _Wrap()
        M.in_channels       = 3 * cfg.latent_ch  # [x_t(2C), Plow(C)]
        M.out_ch            = 2 * cfg.latent_ch  # eps(2C)
        M.ch                = cfg.unet_ch
        M.ch_mult           = cfg.unet_ch_mult
        M.num_res_blocks    = cfg.unet_num_res_blocks
        M.attn_resolutions  = cfg.unet_attn_resolutions
        M.dropout           = cfg.unet_dropout
        M.resamp_with_conv  = cfg.unet_resamp_with_conv
        D.image_size        = cfg.image_size
        self.unet = DiffusionUNet(type("Cfg", (), {"model": M, "data": D})())

        # schedule
        betas = make_beta_schedule(cfg.beta_schedule, cfg.beta_start, cfg.beta_end, cfg.num_diffusion_timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

        # EMA
        self.use_ema = bool(use_ema)
        self.ema_helper = EMAHelper(mu=0.9999) if self.use_ema else None

    # ---- q(x_t|x0) ----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None: noise = torch.randn_like(x0)
        at = self.alphas_cumprod.index_select(0, t).view(-1,1,1,1)
        xt = at.sqrt()*x0 + (1.0 - at).sqrt()*noise
        return xt, noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor):
        at = self.alphas_cumprod.index_select(0, t).view(-1,1,1,1)
        return (xt - (1.0 - at).sqrt()*eps_pred) / at.sqrt()

    # ---- Stage-1 ----
    def stage1(self, I: torch.Tensor) -> Dict[str, torch.Tensor]:
        F = self.encoder(I)
        I_rec = self.decoder(F)
        L = loss_stage1_autoencoder(I, I_rec)
        return {"loss": L, "I_rec": I_rec}

    # ---- Stage-2 ----
    def stage2(self, Ilow: torch.Tensor, Ihigh: torch.Tensor, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            Flow  = self.encoder(Ilow)
            Fhigh = self.encoder(Ihigh)
        Alow,  Plow,  _ = amp_phase(Flow)
        Ahigh, Phigh, _ = amp_phase(Fhigh)

        x0 = phase_to_embedding(Phigh)
        B = x0.size(0)
        if t is None:
            T = self.cfg.num_diffusion_timesteps
            t = torch.randint(0, T, (B,), device=x0.device, dtype=torch.long)
        xt, eps = self.q_sample(x0, t)

        eps_pred = self.unet(torch.cat([xt, Plow], dim=1), t)
        x0_hat = self.predict_x0_from_eps(xt, t, eps_pred)
        P_hat_low = embedding_to_phase(x0_hat)

        A_hat_low = self.lbm(Alow)
        F_prime_low = combine_amp_phase(A_hat_low, P_hat_low)
        if F_prime_low.dtype != Flow.dtype:
            F_prime_low = F_prime_low.to(Flow.dtype)
        F_hat_low = self.lffm(Flow, F_prime_low)
        I_hat_low = self.decoder(F_hat_low)

        L_total, logs = loss_stage2_total(
            eps_pred=eps_pred, eps_gt=eps,
            P_hat_low=P_hat_low, P_high=Phigh,
            A_hat_low=A_hat_low, A_high=Ahigh,
            I_hat_low=I_hat_low, I_high=Ihigh,
            F_hat_low=F_hat_low, F_high=Fhigh,
            lambda1=self.cfg.lambda1, lambda2=self.cfg.lambda2,
        )
        return {"loss": L_total, **logs, "I_hat_low": I_hat_low}

    def sample(self, Ilow: torch.Tensor, S: int = None) -> Dict[str, torch.Tensor]:
        """
        Strict DDPM sampling with S steps (S<=T).
        - If S is None, use full T steps (original behavior).
        - Operates in phase-embedding space conditioned on Plow.
        """
        device = Ilow.device
        Flow = self.encoder(Ilow)
        Alow, Plow, _ = amp_phase(Flow)

        betas = self.betas
        alphas = self.alphas
        abar = self.alphas_cumprod

        T = betas.shape[0]
        # -------- choose S-step schedule (subsequence of [0..T-1]) --------
        if S is None or S >= T:
            seq = torch.arange(T - 1, -1, -1, device=device)  # full T -> T, T-1, ..., 0
        else:
            # 등간격 인덱스 선택 (T개 중 S개). 예: T=1000, S=50 -> 약 20 step 간격
            idx = torch.linspace(0, T - 1, steps=S, device=device).round().long()
            seq = torch.flip(idx, dims=[0])  # 역순으로 내려감

        # abar_prev( t=0일 때는 abar_prev=1 )
        abar_prev = torch.ones_like(abar)
        abar_prev[1:] = abar[:-1]

        B, C, H, W = Flow.shape
        x = torch.randn(B, 2 * C, H, W, device=device)

        for t_scalar in seq.tolist():  # e.g., [999, 979, ..., 0]
            t = torch.full((B,), int(t_scalar), device=device, dtype=torch.long)

            a_t = alphas.index_select(0, t).view(-1, 1, 1, 1)
            abar_t = abar.index_select(0, t).view(-1, 1, 1, 1)
            abar_tm1 = abar_prev.index_select(0, t).view(-1, 1, 1, 1)
            beta_t = betas.index_select(0, t).view(-1, 1, 1, 1)

            # predict eps
            x_in = torch.cat([x, Plow], dim=1)
            eps = self.unet(x_in, t)

            # DDPM mean (eps-parameterization), variance (posterior)
            mean = (1.0 / a_t.sqrt()) * (x - ((1.0 - a_t) / (1.0 - abar_t).sqrt()) * eps)
            var = beta_t * (1.0 - abar_tm1) / (1.0 - abar_t)
            var = torch.clamp(var, min=1e-20)

            if t_scalar > 0:
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean

        P_hat_low = embedding_to_phase(x)
        A_hat_low = self.lbm(Alow)
        F_prime_low = combine_amp_phase(A_hat_low, P_hat_low)
        if F_prime_low.dtype != Flow.dtype:
            F_prime_low = F_prime_low.to(Flow.dtype)
        F_hat_low = self.lffm(Flow, F_prime_low)
        I_hat_low = self.decoder(F_hat_low)

        return {
            "I_hat_low": I_hat_low,
            "F_hat_low": F_hat_low,
            "F_prime_low": F_prime_low,
            "A_hat_low": A_hat_low,
            "P_hat_low": P_hat_low,
        }
