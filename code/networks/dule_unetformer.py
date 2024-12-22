# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
dual unet-former, image was send to transformer on the top, and u-net on the bottom
the transformer's attention is used as a aux crf for u-net

"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math

def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # note that, we already assumed that here, x is B, L, C; L = H * W
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn_ = (q @ k.transpose(-2, -1))

        '''if self.sr_ratio == 1:
            attn_ = attn_ + attn_.permute(0, 1, 3, 2)'''

        attn = (attn_ * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        #####
        #attn_ = attn_.clone().mean(1).reshape(-1, H, W, attn.shape[-1],)
        #attn_ = F.avg_pool2d(attn_.permute(0,3,1,2), kernel_size=self.sr_ratio, stride=self.sr_ratio)
        #attn_ = attn_.reshape(-1, attn.shape[-1], attn.shape[-1])
        attn_copy = attn_.clone().reshape(B, self.num_heads, H, W, attn.shape[-1],)
        if self.sr_ratio > 1:
            attn_copy = F.avg_pool3d(attn_copy, kernel_size=(self.sr_ratio, self.sr_ratio, 1), stride=(self.sr_ratio, self.sr_ratio, 1))
            #attn_copy = attn_copy.reshape(B, self.num_heads, self.sr_ratio, -1, W, attn.shape[-1],).mean(2)
            #attn_copy = attn_copy.reshape(B, self.num_heads, attn_copy.shape[2], self.sr_ratio, -1, attn.shape[-1],).mean(3)
        #print(attn_copy.shape)
        #attn_ = F.avg_pool2d(attn_.permute(0,3,1,2), kernel_size=self.sr_ratio, stride=self.sr_ratio)
        attn_copy = attn_copy.reshape(-1, self.num_heads, attn.shape[-1], attn.shape[-1])
        #####

        return x, attn_copy
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        _x, _attn = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x, _attn



class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EncoderFormer(nn.Module):
    def __init__(self, params):
        super(EncoderFormer, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        self.is_tfm = self.params['is_tfm'] # whether or not the tfm block would ever be computed
        self.is_sep_tfm = self.params['is_sep_tfm'] # whether the tfm head is a seperate one (do not mult. attn back to main stream)

        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])


        # add attention block
        embed_dims = 64
        num_heads = 1
        mlp_ratio = 4
        qkv_bias = False
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path = 0
        norm_layer = nn.LayerNorm
        sr_ratio = 1
        depth = 1 # number of self-attention layers
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer,
            sr_ratio=sr_ratio)
            for i in range(depth)])
        
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn_proj = nn.Conv2d(in_channels=depth * num_heads, out_channels=1, kernel_size=1, bias=True)
        
    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        # <mtian> inject self-attention blocks here                
        x2 = self.down2(x1)
        attn_pred = None
        if self.is_tfm:
            if self.is_sep_tfm:
                # cloning so it does not affect x2!
                x2_tfm = x2.clone()
            else:
                x2_tfm = x2
            ### attention added
            # (Pdb) pp x1.shape
            # torch.Size([12, 32, 128, 128])        
            # (Pdb) pp x2.shape
            # torch.Size([6, 64, 64, 64])
            B, C, H, W = x2_tfm.shape
            attns = []
            # stage 1
            # x, H, W = self.patch_embed1(x)
            x2_tfm = nchw_to_nlc(x2_tfm)
            for i, blk in enumerate(self.block1):
                x2_tfm, attn = blk(x2_tfm, H, W)
                # (Pdb) pp x1.shape
                # torch.Size([6, 16384, 32])
                # (Pdb) pp attn.size()
                # torch.Size([6, 1, 16384, 16384])
                attns.append(attn)
            x2_tfm = self.norm1(x2_tfm)
            x2_tfm = nlc_to_nchw(x2_tfm, (H, W))

            attn_cat = torch.cat(attns, dim=1)#.detach()
            # B, 1, HW, HW
            attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
            attn_pred = self.attn_proj(attn_cat)
            attn_pred = torch.sigmoid(attn_pred).squeeze(1)
            # attn_pred should be B, HW, HW, note that the HW, HW mask could be very sparse given we don't count background vs. background
            # or alternatively we could just incorporate neighboring positions

        ### attention added
        if self.is_tfm and (not self.is_sep_tfm):
            x2 = x2_tfm
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4], attn_pred


class DecoderFormer(nn.Module):
    def __init__(self, params):
        super(DecoderFormer, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x

class UNetFormer(nn.Module):
    def __init__(self, in_chns, class_num, is_tfm=True, is_sep_tfm=False):
        super(UNetFormer, self).__init__()

        params = {
            'in_chns': in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'is_tfm': is_tfm,
            'is_sep_tfm': is_sep_tfm,
        }

        self.encoder = EncoderFormer(params)
        self.decoder = DecoderFormer(params)

    def forward(self, x):
        feature, attn_pred = self.encoder(x)
        output = self.decoder(feature)
        return output, attn_pred

