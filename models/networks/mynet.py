import sys
import math
import torch
import numbers
import numpy as np
import torch.nn as nn
from torch.nn import init
from itertools import repeat
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import collections.abc as container_abcs
from torch.nn.modules.module import Module
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
# from timm.models._efficientnet_blocks import SqueezeExcite as SE
from .restormer_arch import *
from .basic_modules import *
from .newnet import *


def PhiTPhi_fun(x, PhiW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    # 反卷积就是将低分辨率的图像通过0填充成很大的图像再卷积成高分辨率的图像
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        # isinstance 判断一个对象是否是一个已知的类型
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Head(nn.Module):
    def __init__(self, embed_dim=24, drop_path=0.1):
        r""" """
        super().__init__()
        self.embed_dim = embed_dim
        self.block = nn.Sequential(
            nn.Conv2d(1, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, 3, padding=1)
        )
        alpha_0 = 1e-2
        self.alpha = nn.Parameter(
            alpha_0 * torch.ones((1, self.embed_dim, 1, 1)), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.alpha * self.block(x))
        return x


class Tail(nn.Module):
    def __init__(self, embed_dim=24):
        r""" """
        super().__init__()
        self.embed_dim = embed_dim
        self.block = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, 1, 3, padding=1)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class RB(nn.Module):
    def __init__(self, channel, relu_slope, use_HIN=False):
        super(RB, self).__init__()
        self.use_HIN = use_HIN
        if use_HIN:
            self.norm = nn.InstanceNorm2d(channel // 2,
                                          affine=True)  # 与批量归一化（Batch Normalization）类似，但在每个样本的基础上进行归一化，而不是在整个批次上进行
        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out)) + x

        return out

class Unet_Enc(nn.Module):
    def __init__(self,
                 dim_ladder,
                 bias=False):
        super(Unet_Enc, self).__init__()
        self.conv_forward = Head(embed_dim=dim_ladder[0])
        self.enhance1 = TransformerBlock(dim_ladder[0], 2, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.RB1 = RB(dim_ladder[0], 0.2, True)
        self.down1 = nn.Sequential(
            nn.Conv2d(dim_ladder[0], dim_ladder[1] // 4, 3, padding=1, bias=False),
            Rearrange("b c (h t1) (w t2) -> b (c t1 t2) h w", t1=2, t2=2),
        )
        self.enhance2 = TransformerBlock(dim_ladder[1], 4, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.down2 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[2] // 4, 3, padding=1, bias=False),
            Rearrange("b c (h t1) (w t2) -> b (c t1 t2) h w", t1=2, t2=2),
        )
        self.enhance3 = TransformerBlock(dim_ladder[2], 8, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.RB3 = RB(dim_ladder[2], 0.2, True)

    def forward(self, x_in):
        x_level_32 = self.RB1(self.enhance1(self.conv_forward(x_in)))
        x_level_64 = self.RB2(self.enhance2(self.down1(x_level_32)))
        x_level_128 = self.RB3(self.enhance3(self.down2(x_level_64)))
        x_features = [x_level_32, x_level_64, x_level_128]
        return x_features


class Unet_Dec(nn.Module):
    def __init__(self,
                 dim_ladder,
                 bias=False):
        super(Unet_Dec, self).__init__()
        self.RB3 = RB(dim_ladder[2], 0.2, True)
        self.enhance3 = TransformerBlock(dim_ladder[2], 8, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.up2 = nn.Sequential(
            nn.Conv2d(dim_ladder[2], dim_ladder[1] * 4, 3, padding=1, bias=False),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge2 = nn.Conv2d(dim_ladder[1]*2, dim_ladder[1], 1, 1, 0)
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.enhance2 = TransformerBlock(dim_ladder[1], 4, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.up1 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[0] * 4, 3, padding=1, bias=False),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge1 = nn.Conv2d(dim_ladder[0]*2, dim_ladder[0], 1, 1, 0)
        self.RB1 = RB(dim_ladder[0], 0.2, True)
        self.enhance1 = TransformerBlock(dim_ladder[0], 2, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.conv_backward = Tail(embed_dim=dim_ladder[0])

    def forward(self, x_features):
        x_level_64 = self.merge2(torch.cat([self.up2(self.enhance3(self.RB3(x_features[2]))), x_features[1]], dim=1))
        x_level_32 = self.merge1(torch.cat([self.up1(self.enhance2(self.RB2(x_level_64))), x_features[0]], dim=1))
        x_opt = self.conv_backward(self.enhance1(self.RB1(x_level_32)))
        return x_opt
    
class Feature_Dec(nn.Module):
    def __init__(self,
                 dim_ladder,
                 bias=False):
        super(Feature_Dec,self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.1]))
        self.RB3 = RB(dim_ladder[2], 0.2, True)
        self.enhance3 = TransformerBlock(dim_ladder[2], 8, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.up2 = nn.Sequential(
            nn.Conv2d(dim_ladder[2], dim_ladder[1] * 4, 3, padding=1, bias=False),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge2 = nn.Conv2d(dim_ladder[1]*2, dim_ladder[1], 1, 1, 0)
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.enhance2 = TransformerBlock(dim_ladder[1], 4, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.up1 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[0] * 4, 3, padding=1, bias=False),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge1 = nn.Conv2d(dim_ladder[0]*2, dim_ladder[0], 1, 1, 0)
        self.RB1 = RB(dim_ladder[0], 0.2, True)
        self.enhance1 = TransformerBlock(dim_ladder[0], 2, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.conv_backward = Tail(embed_dim=dim_ladder[0])

    def forward(self, x_features):
        x_level_64 = self.merge2(torch.cat([self.up2(self.enhance3(self.RB3(x_features[2]))), x_features[1]], dim=1))
        x_level_32 = self.merge1(torch.cat([self.up1(self.enhance2(self.RB2(x_level_64))), x_features[0]], dim=1))
        x_opt = self.alpha * self.conv_backward(self.enhance1(self.RB1(x_level_32)))
        return x_opt

class Mid_Dec(nn.Module):
    def __init__(self,
                 dim_ladder,
                 bias=False):
        super(Mid_Dec, self).__init__()
        self.RB3 = RB(dim_ladder[2], 0.2, True)
        self.up2 = nn.Sequential(
            nn.Conv2d(dim_ladder[2], dim_ladder[1] * 4, 3, padding=1, bias=bias),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge2 = nn.Conv2d(dim_ladder[1]*2, dim_ladder[1], 1, 1, 0)
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.up1 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[0] * 4, 3, padding=1, bias=bias),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge1 = nn.Conv2d(dim_ladder[0]*2, dim_ladder[0], 1, 1, 0)
        self.RB1 = RB(dim_ladder[0], 0.2, True)

    def forward(self, x_features):
        x_level_128 = self.RB3(x_features[2])
        x_level_64 = self.RB2(self.merge2(torch.cat([self.up2(x_level_128), x_features[1]], dim=1))) + x_features[1]
        x_level_32 = self.RB1(self.merge1(torch.cat([self.up1(x_level_64), x_features[0]], dim=1))) + x_features[0]
        return [x_level_32,x_level_64,x_level_128]

class Mid_Enc(nn.Module):
    def __init__(self,
                 dim_ladder,
                 bias=False):
        super(Mid_Enc,self).__init__()
        self.RB1 = RB(dim_ladder[0], 0.2, True)
        self.down1 = nn.Sequential(
            nn.Conv2d(dim_ladder[0], dim_ladder[1] // 4, 3, padding=1, bias=bias),
            Rearrange("b c (h t1) (w t2) -> b (c t1 t2) h w", t1=2, t2=2),
        )
        self.merge1 = nn.Conv2d(2*dim_ladder[1],dim_ladder[1],1,1,0)
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.down2 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[2] // 4, 3, padding=1, bias=bias),
            Rearrange("b c (h t1) (w t2) -> b (c t1 t2) h w", t1=2, t2=2),
        )
        self.merge2 = nn.Conv2d(2*dim_ladder[2],dim_ladder[2],1,1,0)
        self.RB3 = RB(dim_ladder[2], 0.2, True)

    def forward(self, x_features):
        x_level_32 = self.RB1(x_features[0])
        x_level_64 = self.RB2(self.merge1(torch.cat([self.down1(x_level_32),x_features[1]],dim=1))) + x_features[1]
        x_level_128 = self.RB3(self.merge2(torch.cat([self.down2(x_level_64),x_features[2]],dim=1))) + x_features[2]
        return [x_level_32, x_level_64, x_level_128]

class Mid_Fusion(nn.Module):
    def __init__(self,
                 dim_ladder,
                 bias=False):
        super(Mid_Fusion,self).__init__()

        self.mid_dec = Mid_Dec(dim_ladder=dim_ladder,bias=bias)
        self.mid_enc = Mid_Enc(dim_ladder=dim_ladder,bias=bias)

    def forward(self,x_features):
        x_features = self.mid_enc(self.mid_dec(x_features))
        return x_features

class SWin_Stage(nn.Module):
    def __init__(self, dim, resolution, depth, bias=False):
        super(SWin_Stage, self).__init__()
        self.dim = dim
        self.swin_forward = BasicLayer(dim, (resolution, resolution), depth, 4, 8)
        self.catt_forward = TransformerBlock(dim, dim // 16, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.mid_merge = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1)
        )
        self.soft_thr = nn.Parameter(torch.full((2 * dim, 1, 1), 0.005))
        self.swin_backward = BasicLayer(dim, (resolution, resolution), depth, 4, 8)
        self.catt_backward = TransformerBlock(dim, dim // 16, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.merge = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2 * dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x_feature):
        x_in_swin = self.swin_forward(x_feature)
        x_in_catt = self.catt_forward(x_feature)
        x_in = self.mid_merge(torch.cat([x_in_swin, x_in_catt], dim=1))
        x_out = torch.mul(torch.sign(x_in), F.relu(torch.abs(x_in) - self.soft_thr))
        x_out_swin, x_out_catt = torch.split(x_out, split_size_or_sections=self.dim, dim=1)
        x_opt_1 = self.swin_backward(x_out_swin) + x_feature
        x_opt_2 = self.catt_backward(x_out_catt) + x_feature
        x_opt = self.merge(torch.cat([x_opt_1, x_opt_2], dim=1))
        return x_opt

class Denoise_Block(nn.Module):
    def __init__(self,
                 dim_ladder,
                 resolution,
                 bias=False):
        super(Denoise_Block, self).__init__()
        self.r = [1, 2, 4]
        self.nf = [dim_ladder[0], dim_ladder[1], dim_ladder[2]]
        self.Grad1 = nn.Sequential(
            nn.Conv2d(self.r[0] ** 2 + 2*self.nf[0], self.nf[0], 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.nf[0],self.nf[0],3,padding=1),
            TransformerBlock(self.nf[0], self.nf[0] // 16, 2.66, bias=bias, LayerNorm_type="WithBias"),
            nn.Sigmoid()
        )
        self.swin_thr1 = SWin_Stage(self.nf[0], resolution, 2)

        self.Grad2 = nn.Sequential(
            nn.Conv2d(self.r[1] ** 2 + 2*self.nf[1], self.nf[1], 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.nf[1], self.nf[1], 3, padding=1),
            TransformerBlock(self.nf[1], self.nf[1] // 16, 2.66, bias=bias, LayerNorm_type="WithBias"),
            nn.Sigmoid()
        )
        self.swin_thr2 = SWin_Stage(self.nf[1], resolution // 2, 2)

        self.Grad3 = nn.Sequential(
            nn.Conv2d(self.r[2] ** 2 + 2*self.nf[2], self.nf[2], 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.nf[2], self.nf[2], 3, padding=1),
            TransformerBlock(self.nf[2], self.nf[2] // 16, 2.66, bias=bias, LayerNorm_type="WithBias"),
            nn.Sigmoid()
        )
        self.swin_thr3 = SWin_Stage(self.nf[2], resolution // 4, 2)

    def forward(self, x_features, Phiweight, PhiTb):
        x_1 = F.pixel_shuffle(x_features[0], upscale_factor=self.r[0])
        b, c, h, w = x_1.shape
        PhiT_Phi_x = PhiTPhi_fun(x_1.reshape(-1, 1, h, w), Phiweight).reshape(b, c, h, w)
        PhiT_Phi_x = F.pixel_unshuffle(PhiT_Phi_x, downscale_factor=self.r[0])
        grad_1 = self.Grad1(
            torch.cat([x_features[0], PhiT_Phi_x, F.pixel_unshuffle(PhiTb, self.r[0])], dim=1))
        x_features[0] = x_features[0] - grad_1
        x_features[0] = self.swin_thr1(x_features[0])

        x_2 = F.pixel_shuffle(x_features[1], upscale_factor=self.r[1])
        b, c, h, w = x_2.shape
        PhiT_Phi_x = PhiTPhi_fun(x_2.reshape(-1, 1, h, w), Phiweight).reshape(b, c, h, w)
        PhiT_Phi_x = F.pixel_unshuffle(PhiT_Phi_x, downscale_factor=self.r[1])
        grad_2 = self.Grad2(
            torch.cat([x_features[1], PhiT_Phi_x, F.pixel_unshuffle(PhiTb, self.r[1])], dim=1))
        x_features[1] = x_features[1] - grad_2
        x_features[1] = self.swin_thr2(x_features[1])

        x_3 = F.pixel_shuffle(x_features[2], upscale_factor=self.r[2])
        b, c, h, w = x_3.shape
        PhiT_Phi_x = PhiTPhi_fun(x_3.reshape(-1, 1, h, w), Phiweight).reshape(b, c, h, w)
        PhiT_Phi_x = F.pixel_unshuffle(PhiT_Phi_x, downscale_factor=self.r[2])
        grad_3 = self.Grad3(
            torch.cat([x_features[2], PhiT_Phi_x, F.pixel_unshuffle(PhiTb, self.r[2])], dim=1))
        x_features[2] = x_features[2] - grad_3
        x_features[2] = self.swin_thr3(x_features[2])

        return x_features

class PMD_Net(nn.Module):
    def __init__(self, layer_num=7, resolution=64, rate=10):
        super(PMD_Net, self).__init__()

        self.patch_size = resolution
        self.n_input = int(rate * 0.01 * (resolution**2))
        self.layer_num = layer_num

        self.Phiweight = nn.Parameter(
            init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))

        dim_ladder = [32, 48, 64]

        self.Encoder = Unet_Enc(dim_ladder=dim_ladder)

        block_list = []
        for i in range(0, self.layer_num-1):
            block_list.append(Denoise_Block(dim_ladder=dim_ladder, resolution=resolution))
            block_list.append(Mid_Fusion(dim_ladder=dim_ladder))
        block_list.append(Denoise_Block(dim_ladder=dim_ladder, resolution=resolution))
        self.denoise_stage = nn.ModuleList(block_list)

        self.Decoder = Unet_Dec(dim_ladder=dim_ladder)
        self.Features_Dec = Feature_Dec(dim_ladder=dim_ladder)

    def forward(self, input):  # shape (batch_size, C=1, H=96, W=96)

        Phix = F.conv2d(input, self.Phiweight, stride=self.patch_size, padding=0, bias=None)  # 64*10*3*3
        # 反卷积可以认为先对feature map进行插值/padding操作得到新的feature map然后进行常规的卷积运算
        PhiTb = F.conv_transpose2d(Phix, self.Phiweight, stride=self.patch_size)

        x = PhiTb
        x_features = self.Encoder(x)
        shotcut = [torch.zeros_like(x_features[i]) for i in range(len(x_features))]
        for i in range(0, self.layer_num-1):
            x_features = self.denoise_stage[2*i](x_features, self.Phiweight, PhiTb)
            for j in range(len(shotcut)):
                shotcut[j] = shotcut[j] + x_features[j]
            x_features = self.denoise_stage[2*i+1](x_features)
        x_features = self.denoise_stage[2*(self.layer_num-1)](x_features, self.Phiweight, PhiTb)
        for j in range(len(shotcut)):
                shotcut[j] = shotcut[j] + x_features[j]
        x_opt = self.Decoder(x_features) + self.Features_Dec(shotcut)

        return x_opt
