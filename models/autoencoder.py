# models/encdec_lighten.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---- 가벼운 빌딩블록 (원 코드 스타일 유지) -----------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, 1, 1)
        self.act   = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip  = nn.Conv2d(in_ch,  out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.skip(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, output_padding=1)
        self.act    = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        return self.act(self.deconv(x))

# ---- 피처 피라미드(인코더전용) -----------------------------------
class FeaturePyramid(nn.Module):
    """
    입력: (B,3,H,W)  출력: level2 (B, 4*base_ch, H/8, W/8)
    원본 feature_pyramid(level0/1/2) 중 level2만 latent로 사용.
    """
    def __init__(self, base_ch: int):
        super().__init__()
        ch = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch, 5, 1, 2),
            nn.Conv2d(ch, ch, 5, 1, 2),
        )
        self.block0 = ResBlock(ch, ch)
        self.down0  = nn.Conv2d(ch, ch, 3, 2, 1)          # H/2

        self.block1 = ResBlock(ch, 2*ch)
        self.down1  = nn.Conv2d(2*ch, 2*ch, 3, 2, 1)      # H/4

        self.block2 = ResBlock(2*ch, 4*ch)
        self.down2  = nn.Conv2d(4*ch, 4*ch, 3, 2, 1)      # H/8

    def forward(self, x):
        h0 = self.block0(self.stem(x))
        h1 = self.block1(self.down0(h0))
        h2 = self.block2(self.down1(h1))
        z  = self.down2(h2)  # (B, 4*base_ch, H/8, W/8)
        return z

# ---- 라이트닝용 인코더/디코더 (Retinex 완전 제거) -----------------
class LightenEncoder(nn.Module):
    """
    이미지 -> 라텐트 z
    latent_ch를 지정하면 1x1로 채널을 맞춰줌. (기본: 4*base_ch)
    """
    def __init__(self, base_ch: int = 64, latent_ch: int = None):
        super().__init__()
        self.base_ch   = base_ch
        self.fpn       = FeaturePyramid(base_ch)
        in_latent_ch   = 4*base_ch
        out_latent_ch  = in_latent_ch if latent_ch is None else latent_ch
        self.proj = nn.Identity() if out_latent_ch == in_latent_ch else nn.Conv2d(in_latent_ch, out_latent_ch, 1, 1, 0)
        self.latent_ch = out_latent_ch

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fpn(x)          # (B, 4*base_ch, H/8, W/8)
        z = self.proj(z)         # (B, latent_ch,  H/8, W/8)
        return z

class LightenDecoder(nn.Module):
    """
    z -> 이미지
    ReconNet의 업샘플 경로를 '스킵 없이' 미러 구성.
    입력 z 채널=latent_ch (기본 4*base_ch).
    출력은 [-1,1]을 선호하면 tanh, [0,1] 쓰면 sigmoid로.
    """
    def __init__(self, base_ch: int = 64, latent_ch: int = None, out_act: str = "tanh"):
        super().__init__()
        self.base_ch  = base_ch
        in_ch = latent_ch if latent_ch is not None else 4*base_ch

        # H/8 -> H/4
        self.up0   = Up(in_ch, 2*base_ch)
        self.rb0_0 = ResBlock(2*base_ch, 2*base_ch)
        self.rb0_1 = ResBlock(2*base_ch, 2*base_ch)

        # H/4 -> H/2
        self.up1   = Up(2*base_ch, base_ch)
        self.rb1_0 = ResBlock(base_ch, base_ch)
        self.rb1_1 = ResBlock(base_ch, base_ch)

        # H/2 -> H
        self.up2   = Up(base_ch, base_ch)
        self.rb2_0 = ResBlock(base_ch, base_ch)
        self.rb2_1 = ResBlock(base_ch, base_ch)

        self.conv2 = nn.Conv2d(base_ch, base_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(base_ch, 3, 1, 1, 0)
        self.act   = nn.LeakyReLU(inplace=True)
        self.out_act = out_act

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.rb0_1(self.rb0_0(self.up0(z)))
        h = self.rb1_1(self.rb1_0(self.up1(h)))
        h = self.rb2_1(self.rb2_0(self.up2(h)))
        h = self.act(self.conv2(h))
        x = self.conv3(h)
        if self.out_act == "tanh":
            x = torch.tanh(x)       # [-1,1]
        elif self.out_act == "sigmoid":
            x = torch.sigmoid(x)    # [0,1]
        return x
