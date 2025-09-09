# unet.py
# ------------------------------------------------------------
# Diffusion U-Net for Latent Fourier Phase Diffusion (robust)
# - input:  x_in = cat([x_t(2C), P_low(C)], dim=1)  => in_channels = 3*C
# - output: eps_pred with channels = 2*C
# - timestep t : LongTensor [B], classic DDPM sinusoidal embedding
# - up path uses AlignedResBlock (runtime 1x1 align) to prevent channel mismatch
# ------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Positional (timestep) embedding
# -------------------------
def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int):
    """
    Sinusoidal timestep embedding (DDPM standard).
    timesteps: LongTensor [B]
    return:   FloatTensor [B, embedding_dim]
    """
    assert timesteps.dim() == 1
    half_dim = embedding_dim // 2
    scale = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, 2*half_dim]
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def Normalize(c):
    # num_groups=32 is typical; if c < 32, PyTorch will throw. But here c is >= 64.
    return nn.GroupNorm(num_groups=32, num_channels=c, eps=1e-6, affine=True)


# -------------------------
# Building blocks
# -------------------------
class ResnetBlock(nn.Module):
    """
    GN -> SiLU -> 3x3 -> +temb -> GN -> SiLU -> Drop -> 3x3  (+ skip)
    """
    def __init__(self, in_channels, out_channels=None, temb_channels=512, dropout=0.0, use_conv_shortcut=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_conv_shortcut = use_conv_shortcut and (self.in_channels != self.out_channels)

        self.norm1 = Normalize(self.in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)

        self.temb_proj = nn.Linear(temb_channels, self.out_channels)

        self.norm2 = Normalize(self.out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.skip = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
            else:
                self.skip = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        else:
            self.skip = nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.drop(self.act2(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """
    Self-attention in space: 1x1 q/k/v + softmax(qk^T/sqrt(c))v + 1x1 out
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = Normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).transpose(1, 2)  # [B,HW,C]
        k = k.reshape(B, C, H * W)                  # [B,C,HW]
        w = torch.bmm(q, k) * (C ** -0.5)           # [B,HW,HW]
        w = torch.softmax(w, dim=-1)
        v = v.reshape(B, C, H * W).transpose(1, 2)  # [B,HW,C]
        h = torch.bmm(w, v).transpose(1, 2).reshape(B, C, H, W)
        h = self.proj(h)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.op = nn.Conv2d(channels, channels, 3, 2, 1)
        else:
            self.op = nn.AvgPool2d(2)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        self.up = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1) if with_conv else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


# -------------------------
# Runtime-aligned ResBlock (fixes channel mismatch)
# -------------------------
class AlignedResBlock(nn.Module):
    """
    Wraps ResnetBlock to auto-align input channels via a 1x1 conv
    if the concatenated input channels != expected_in_ch.
    """
    def __init__(self, expected_in_ch, out_ch, temb_channels=512, dropout=0.0):
        super().__init__()
        self.expected_in_ch = int(expected_in_ch)
        self.out_ch = int(out_ch)
        self.block = ResnetBlock(self.expected_in_ch, self.out_ch,
                                 temb_channels=temb_channels,
                                 dropout=dropout,
                                 use_conv_shortcut=True)
        self.align = None  # lazily created nn.Conv2d(in_actual, expected_in_ch, 1)

    def forward(self, x, temb):
        in_ch = x.shape[1]
        if in_ch != self.expected_in_ch:
            if (self.align is None) or (self.align.in_channels != in_ch):
                # create / recreate align conv lazily with correct in_ch
                self.align = nn.Conv2d(in_ch, self.expected_in_ch, kernel_size=1, stride=1, padding=0).to(x.device)
            x = self.align(x)
        return self.block(x, temb)


# -------------------------
# Diffusion U-Net
# -------------------------
class DiffusionUNet(nn.Module):
    """
    Config keys required:
      model.in_channels   = 3*C         # cat([x_t(2C), P_low(C)])
      model.out_ch        = 2*C         # eps pred
      model.ch            = base width  # e.g., 64
      model.ch_mult       = (1,2,4,4)   # pyramids
      model.num_res_blocks
      model.attn_resolutions  # e.g., (16,)
      model.dropout
      model.resamp_with_conv  # True
      data.image_size         # input spatial size to UNet (latent size)

    forward(x, t):
      x: [B, in_channels, h, w]
      t: [B] (int)
    """
    def __init__(self, config):
        super().__init__()
        M = config.model
        D = config.data

        self.in_channels = int(M.in_channels)    # expect 3*C
        self.out_ch = int(M.out_ch)              # expect 2*C
        self.ch = int(M.ch)
        self.num_resolutions = len(tuple(M.ch_mult))
        self.num_res_blocks = int(M.num_res_blocks)
        self.attn_resolutions = set(int(a) for a in tuple(M.attn_resolutions))
        self.dropout = float(M.dropout)
        self.resamp_with_conv = bool(M.resamp_with_conv)
        self.image_size = int(D.image_size)

        # timestep embedding
        self.temb_ch = 4 * self.ch
        self.temb = nn.Sequential(
            nn.Linear(self.ch, self.temb_ch),
            nn.SiLU(inplace=True),
            nn.Linear(self.temb_ch, self.temb_ch),
        )

        # in conv
        self.conv_in = nn.Conv2d(self.in_channels, self.ch, 3, 1, 1)

        # ----------------------
        # Down path + record exact skip channels
        # ----------------------
        curr_res = self.image_size
        ch_mult = (1,) + tuple(M.ch_mult)  # prepend base 1
        self.down = nn.ModuleList()

        # The forward will push into hs: first conv_in(x), then after each resblock,
        # and after each downsample (except last level). We mirror that here by recording
        # the channel counts that will be concatenated during up path.
        self._skip_chans = []
        # first pushed: conv_in output
        self._skip_chans.append(self.ch)

        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * ch_mult[i_level]
            block_out = self.ch * M.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, temb_channels=self.temb_ch, dropout=self.dropout))
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(block_in))
                # after each block forward we will push skip (curr), so record channels
                self._skip_chans.append(block_in)
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=self.resamp_with_conv)
                curr_res //= 2
                # after downsample forward we will push skip, record its channels
                self._skip_chans.append(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block1 = ResnetBlock(block_in, block_in, temb_channels=self.temb_ch, dropout=self.dropout)
        self.mid.attn1 = AttnBlock(block_in)
        self.mid.block2 = ResnetBlock(block_in, block_in, temb_channels=self.temb_ch, dropout=self.dropout)

        # ----------------------
        # Up path — use AlignedResBlock to guard channel mismatch
        # ----------------------
        self.up = nn.ModuleList()
        skip_chans_build = list(self._skip_chans)  # copy for construction
        # current feature channels entering the first up block = block_in (from mid)
        up_block_in = block_in
        # note: curr_res at this point equals the last downsampled resolution

        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch * M.ch_mult[i_level]

            # num_res_blocks + 1 blocks per level (mirrors common DDPM UNet)
            for _ in range(self.num_res_blocks + 1):
                assert len(skip_chans_build) > 0, "skip list underflow: mis-specified down/up schedule"
                skip_in = skip_chans_build.pop()           # channels of the skip that will be concatenated
                expected_in = up_block_in + skip_in        # concat channels expected by this block
                block.append(AlignedResBlock(expected_in_ch=expected_in,
                                             out_ch=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=self.dropout))
                up_block_in = block_out
                # attn at this resolution if requested
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(up_block_in))

            up_level = nn.Module()
            up_level.block = block
            up_level.attn = attn
            if i_level != 0:
                up_level.upsample = Upsample(up_block_in, with_conv=self.resamp_with_conv)
                curr_res *= 2
            self.up.append(up_level)

        # out
        self.norm_out = Normalize(up_block_in)
        self.act = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(up_block_in, self.out_ch, 3, 1, 1)

    def forward(self, x, t):
        """
        x: [B, in_channels, h, w]  (in_channels = 3*C)
        t: [B] (long)
        """
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)  # [B, ch]
        temb = self.temb(temb)                     # [B, 4*ch]

        # down
        hs = [self.conv_in(x)]
        curr = hs[-1]
        for i_level, down in enumerate(self.down):
            for i_block, block in enumerate(down.block):
                curr = block(curr, temb)
                if len(down.attn) > 0:
                    curr = down.attn[i_block](curr)
                hs.append(curr)  # record skip
            if hasattr(down, "downsample"):
                curr = down.downsample(curr)
                hs.append(curr)  # record skip after downsample

        # mid
        h = self.mid.block1(curr, temb)
        h = self.mid.attn1(h)
        h = self.mid.block2(h, temb)

        # up
        for i_level, up in enumerate(self.up):
            for i_block, block in enumerate(up.block):
                # concat with the exact recorded skip
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, temb)
                if len(up.attn) > 0:
                    h = up.attn[i_block](h)
            if hasattr(up, "upsample"):
                h = up.upsample(h)

        # out
        h = self.act(self.norm_out(h))
        return self.conv_out(h)
