import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 네가 제시한 ReLU+InstanceNorm ResBlock 그대로
# -------------------------
class resBlock(nn.Module):
    def __init__(self, channelDepth, windowSize=3):
        super(resBlock, self).__init__()
        padding = math.floor(windowSize/2)
        self.conv1 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, padding, bias=True)
        self.conv2 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, padding, bias=True)
        self.IN_conv = nn.InstanceNorm2d(channelDepth, track_running_stats=False, affine=False)

    def forward(self, x):
        res = x
        x = F.relu(self.IN_conv(self.conv1(x)))
        x = self.IN_conv(self.conv2(x))
        x = F.relu(x + res)
        return x

# -------------------------
# 공통 유틸 (논문: Q,K,V는 3×3 depth-wise conv, C×C 채널 어텐션)
# -------------------------
class DepthwiseConv3x3(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=True)
    def forward(self, x): return self.dw(x)

def l2_normalize(x, dim=-1, eps=1e-6):
    return x / (x.pow(2).sum(dim=dim, keepdim=True).clamp_min(eps).sqrt())

class ChannelAttention_CxC(nn.Module):
    """
    Q,K,V: [B,C,H,W]  ->  flatten HW -> L2 normalize(Q,K) -> A = Q @ K^T (B,C,C)
                          O = A @ V_flat -> reshape to [B,C,H,W]
    (논문 서술대로 dot-product; 별도의 softmax/스케일/게이트 없음)
    """
    def forward(self, Q, K, V):
        B, C, H, W = Q.shape
        HW = H * W
        Qf = Q.view(B, C, HW)
        Kf = K.view(B, C, HW)
        Vf = V.view(B, C, HW)
        Qn = l2_normalize(Qf, dim=-1)
        Kn = l2_normalize(Kf, dim=-1)
        A  = torch.bmm(Qn, Kn.transpose(1, 2))   # [B,C,C]
        Of = torch.bmm(A, Vf)                    # [B,C,HW]
        return Of.view(B, C, H, W)

# -------------------------
# LBM (원문 재현): RB(Alow) → DW로 Q,K,V → 채널 어텐션 → 1×1 conv → ResBlock → Â_low
# -------------------------
class LBM_Original_ReLU(nn.Module):
    def __init__(self, channels, num_pre_res=2, num_post_res=1, Block=resBlock):
        super().__init__()
        self.pre = nn.Sequential(*[Block(channels) for _ in range(num_pre_res)])
        self.dw_q = DepthwiseConv3x3(channels)
        self.dw_k = DepthwiseConv3x3(channels)
        self.dw_v = DepthwiseConv3x3(channels)
        self.attn = ChannelAttention_CxC()
        self.fuse_1x1 = nn.Conv2d(channels, channels, 1, bias=True)
        self.post_rb = nn.Sequential(*[Block(channels) for _ in range(num_post_res)])

    def forward(self, A_low):                  # (B,C,H,W)
        x = self.pre(A_low)                    # RB(Alow)
        Q = self.dw_q(x); K = self.dw_k(x); V = self.dw_v(x)
        O = self.attn(Q, K, V)                 # 채널 self-attn (C×C)
        y = self.fuse_1x1(O)                   # 1×1
        A_hat = self.post_rb(y)                # ResBlock 통과 (최종 출력)
        return A_hat

# -------------------------
# LFFM (원문 재현): Q=RB(DW(F′)) , K,V=DW(Flow) → cross 채널 어텐션 → ResBlock → F̂_low
# -------------------------
class LFFM_Original_ReLU(nn.Module):
    def __init__(self, channels, q_res=1, out_res=1, Block=resBlock):
        super().__init__()
        self.q_dw = DepthwiseConv3x3(channels)
        self.q_rb = nn.Sequential(*[Block(channels) for _ in range(q_res)])
        self.k_dw = DepthwiseConv3x3(channels)
        self.v_dw = DepthwiseConv3x3(channels)
        self.attn = ChannelAttention_CxC()
        self.post = nn.Sequential(*[Block(channels) for _ in range(out_res)])

    def forward(self, Flow, Fprime_low):       # (B,C,H,W), (B,C,H,W)
        Q = self.q_rb(self.q_dw(Fprime_low))   # Q = RB(DW(F′_low))
        K = self.k_dw(Flow)                    # K = DW(Flow)
        V = self.v_dw(Flow)                    # V = DW(Flow)
        O = self.attn(Q, K, V)                 # cross 채널 어텐션 (C×C)
        F_hat = self.post(O)                   # ResBlock 통과 (최종 출력)
        return F_hat
