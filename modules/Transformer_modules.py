import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange





# ---------- Basic blocks ----------
class Depth_conv(nn.Module):
    def __init__(self, in_ch, bias=True):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=in_ch,
            kernel_size=3, stride=1, padding=1,
            groups=in_ch, bias=bias
        )
    def forward(self, x): return self.depth_conv(x)

class one_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
    def forward(self, x): return self.pw(x)
class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()

        sequence = []

        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]

        self.model = nn.Sequential(*sequence)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        out = self.model(x) + self.conv(x)

        return out

class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.q_dw_con = Depth_conv(dim,bias=bias)
        self.k_dw_con = Depth_conv(dim,bias=bias)
        self.v_dw_con = Depth_conv(dim,bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q=self.q_dw_con(x)
        k=self.k_dw_con(x)
        v=self.v_dw_con(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.q_dw_con = Depth_conv(dim,bias=bias)
        self.k_dw_con = Depth_conv(dim,bias=bias)
        self.v_dw_con = Depth_conv(dim,bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)
        self.res_block=Res_block(in_channels=dim, out_channels=dim)


    def forward(self, hidden_states, ctx):
        b, c, h, w = hidden_states.shape
        Query=self.res_block(hidden_states)
        q=self.q_dw_con(Query)
        k=self.k_dw_con(ctx)
        v=self.v_dw_con(ctx)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)


        return out

class LBM(nn.Module):
    def __init__(self,channels):
        super(LBM,self).__init__()

        self.attn=Self_Attention(dim=channels,num_heads=1,bias=True)
        self.res_block=Res_block(in_channels=channels, out_channels=channels)

    def forward(self, x) :
        x=self.res_block(x)
        x=self.attn(x)
        x=self.res_block(x)

        return x
class LFFM(nn.Module):
    def __init__(self,channels):
        super(LFFM,self).__init__()
        self.res_blcok=Res_block(in_channels=channels,out_channels=channels)
        self.cross_attn=Cross_Attention(dim= channels, num_heads=1,bias=True)
    def forward(self, query,key):
        x=self.cross_attn(query,key)
        out=self.res_blcok(x)

        return out
