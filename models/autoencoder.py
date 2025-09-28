import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.Transformer_modules import LBM,LFFM
# =========================
# Safe / reusable FFT modules
# (paper: 2D-FFT on encoder features; amplitude=illumination, phase=details)
# =========================

def _to_float32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype in (torch.float32, torch.float64) else x.float()

def _cast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == ref.dtype else x.to(ref.dtype)

class FFT2d(nn.Module):
    """
    Differentiable 2D FFT that returns (amplitude, phase).
    - No fftshift (paper doesn't use it).
    - Norm follows torch.fft.fft2; default "backward" as in paper setting.
    """
    def __init__(self, norm: str = "backward", eps: float = 1e-8, keep_dtype: bool = True):
        super().__init__()
        self.norm = norm
        self.eps = eps
        self.keep_dtype = keep_dtype

    def forward(self, x: torch.Tensor):
        # x: [B,C,H,W], real
        xin = _to_float32(x)
        z = torch.fft.fft2(xin, norm=self.norm)             # complex64/128
        amp = torch.abs(z).clamp_min(self.eps)              # [B,C,H,W]
        pha = torch.atan2(z.imag, z.real)                   # [-pi, pi]
        if self.keep_dtype:
            amp = _cast_like(amp, x)
            pha = _cast_like(pha, x)
        return amp, pha

class IFFT2d(nn.Module):
    """
    Inverse of FFT2d: (amplitude, phase) -> real reconstruction (last two dims).
    """
    def __init__(self, norm: str = "backward", keep_dtype: bool = True):
        super().__init__()
        self.norm = norm
        self.keep_dtype = keep_dtype

    def forward(self, amp: torch.Tensor, pha: torch.Tensor):
        # amp, pha: [B,C,H,W]
        ref = amp
        amp32 = _to_float32(amp)
        pha32 = _to_float32(pha)
        real = amp32 * torch.cos(pha32)
        imag = amp32 * torch.sin(pha32)
        rec = torch.fft.ifft2(torch.complex(real, imag), norm=self.norm).real  # [B,C,H,W]
        return _cast_like(rec, ref) if self.keep_dtype else rec


# =========================
# Cleaned building blocks (fixed typos)
# =========================

class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.point(self.depth(x))

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)

class UpSampling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.deconv(x))

class ChannelDown(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # reduce 4C -> 2C (feature compression before FFT / Retinex / etc.)
        self.conv = nn.Conv2d(channels * 4, channels * 2, 3, 1, 1)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))

class ChannelUp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(3, channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(channels, channels * 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, 3, 1, 1)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x

class FeaturePyramid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 5, 1, 2),
            nn.Conv2d(channels, channels, 5, 1, 2),
        )
        self.block0 = ResBlock(channels, channels)
        self.down0  = nn.Conv2d(channels, channels, 3, 2, 1)          # /2
        self.block1 = ResBlock(channels, channels * 2)
        self.down1  = nn.Conv2d(channels * 2, channels * 2, 3, 2, 1)   # /4
        self.block2 = ResBlock(channels * 2, channels * 4)
        self.down2  = nn.Conv2d(channels * 4, channels * 4, 3, 2, 1)   # /8

    def forward(self, x):
        l0 = self.down0(self.block0(self.stem(x)))     # C, H/2,  W/2
        l1 = self.down1(self.block1(l0))               # 2C,H/4,  W/4
        l2 = self.down2(self.block2(l1))               # 4C,H/8,  W/8
        return l0, l1, l2

class ReconNet(nn.Module):
    """
    - Input x: [B, 6, H, W] = concat([low_rgb(3), high_rgb(3)]) per paper's paired setting.
    - When pred_fea is None: acts as encoder, returning (low_fea_down8, high_fea_down8).
    - When pred_fea is given: acts as decoder to predict image from pred_fea and skip connections.
    """
    def __init__(self, channels):
        super().__init__()
        self.pyramid = FeaturePyramid(channels)
        self.ch_down = ChannelDown(channels)
        self.ch_up   = ChannelUp(channels)

        self.block_up0 = ResBlock(channels * 4, channels * 4)
        self.block_up1 = ResBlock(channels * 4, channels * 4)
        self.up0 = UpSampling(channels * 4, channels * 2)

        self.block_up2 = ResBlock(channels * 2, channels * 2)
        self.block_up3 = ResBlock(channels * 2, channels * 2)
        self.up1 = UpSampling(channels * 2, channels)

        self.block_up4 = ResBlock(channels, channels)
        self.block_up5 = ResBlock(channels, channels)
        self.up2 = UpSampling(channels, channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, 3, 1, 1, 0)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x, pred_fea=None):
        if pred_fea is None:
            low_l2, low_l4, low_l8   = self.pyramid(x[:, :3, ...])
            high_l2, high_l4, high_l8 = self.pyramid(x[:, 3:, ...])
            low_fea_down8  = self.ch_down(low_l8)
            high_fea_down8 = self.ch_down(high_l8)
            return low_fea_down8, high_fea_down8

        else:# decoder path (use low branch skips)
            low_l2, low_l4, low_l8 = self.pyramid(x[:, :3, ...])  # only low image for skips
            pred_fea = self.ch_up(pred_fea)                            # expand channels: 3 -> 4C

            pred_fea2 = self.block_up0(pred_fea)
            pred_fea2 = self.block_up1(pred_fea2 + low_l8)        # fuse /8
            pred_fea2 = self.up0(pred_fea2)

            pred_fea4 = self.block_up2(pred_fea2)
            pred_fea4 = self.block_up3(pred_fea4 + low_l4)        # fuse /4
            pred_fea4 = self.up1(pred_fea4)

            pred_fea8 = self.block_up4(pred_fea4)
            pred_fea8 = self.block_up5(pred_fea8 + low_l2)        # fuse /2
            pred_fea8= self.up2(pred_fea8)

            out = self.act(self.conv2(pred_fea8))
            pred_img = self.conv3(out)
            return pred_img


# =========================
# Fourier Decomposer wrapper
# (Stage-1: you may train only E/D; this module simply exposes A,P if needed)
# =========================

class FFTDecomposer(nn.Module):
    """
    Wraps ReconNet (encoder role) + FFT2d/IFFT2d.
    - forward(images, pred_fea=None)
      * pred_fea is None: returns low/high latent features and their (A,P)
      * pred_fea is not None: decodes pred_fea into an image using ReconNet
    """
    def __init__(self, channels=64):
        super().__init__()
        self.base_channels=channels
        self.recon = ReconNet(channels)
        self.adapter_to3=nn.Conv2d(in_channels=channels *2, out_channels=3,kernel_size=1,bias=True)

    def forward(self, images: torch.Tensor, pred_fea: torch.Tensor = None):
        out = {}
        if pred_fea is None:
            low_fea_d8, high_fea_d8 = self.recon(images, pred_fea=None)      # [B, 2C, H/8, W/8]
            low_fft  = torch.fft.fft2(low_fea_d8,norm='ortho')                           # amplitude/phase (paper Fig.2)
            high_fft  = torch.fft.fft2(high_fea_d8,norm='ortho')
            low_A=torch.abs(low_fft)
            low_P=torch.angle(low_fft)
            high_A=torch.abs(high_fft)
            high_P=torch.angle(high_fft)
            low_P=torch.cat([torch.cos(low_P),torch.sin(low_P)],dim=1)
            high_P=torch.cat([torch.cos(high_P),torch.sin(high_P)],dim=1)
            out["I_high"]=images[:,3:6,:,:]
            out["low_A"]=low_A
            out["low_P"]=low_P
            out["high_A"]=high_A
            out["high_P"]=high_P
            out["low_fea"]=low_fea_d8
            out["high_fea"]=high_fea_d8
        else:

            # pred_fea should be 3-channel map before ChannelUp; adapt as you used in training pipeline.
            if pred_fea.shape[1] !=3:
                pred_fea=self.adapter_to3(pred_fea)
            pred_img = self.recon(images[:,:3,...], pred_fea=pred_fea)
            out["pred_img"] = pred_img
        return out
    def to3(self,z):
        return self.adapter_to3(z)
    # handy helper to reconstruct latent back from (A,P) — useful for ablations
    def reconstruct_latent(self, A: torch.Tensor, P: torch.Tensor):
        return self.torch.fft.ifft2(A, P).real
