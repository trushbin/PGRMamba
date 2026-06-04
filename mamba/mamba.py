import torch.nn as nn
import torch
from functools import partial
from einops import rearrange
import sys
import os
# Add local mamba path to sys.path to prioritize local mamba_ssm
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from mamba_ssm.modules.mamba_simple import Mamba
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv3D(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, in_ch, out_ch, k, s=1, p=None, g=1, d=1, act=None):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, autopad(k, p, d), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        if act is None:
            self.act = nn.ReLU()
        else:
            self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # Reshape weight and bias to (1, C, 1, 1, ...) for broadcasting
            shape = [1, -1] + [1] * (x.dim() - 2)
            weight = self.weight.view(*shape)
            bias = self.bias.view(*shape)
            x = weight * x + bias
            return x



class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x





############################## 训练使用 ######################################
class RegMamba(nn.Module):
    def __init__(self, dim, dim_shallow, d_state=16, d_conv=4, expand=2, keep_ratio=0.1): 
        super().__init__()
        self.dim = dim
        self.keep_ratio = keep_ratio  
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False
        )
        
        self.dog_conv = nn.Conv3d(dim_shallow, dim_shallow, kernel_size=5, 
                                  padding=5//2, groups=dim_shallow, bias=False)
        self._init_dog_weights(self.dog_conv, dim_shallow, sigma1=0.8, sigma2=1.6)

        self.gate_beta = nn.Parameter(torch.zeros(1))  # 控制深层信息
        self.gate_dog = nn.Parameter(torch.zeros(1))   # 控制DoG信息

        self.sigmoid = nn.Sigmoid()

    def _init_dog_weights(self, conv, channels, sigma1, sigma2):
        size = 5
        ax = torch.linspace(-(size-1)/2., (size-1)/2., size)
        zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing='ij')
        dist_sq = xx**2 + yy**2 + zz**2
        
        g1 = torch.exp(-0.5 * dist_sq / sigma1**2)
        g1 /= (g1.sum() + 1e-8)
        g2 = torch.exp(-0.5 * dist_sq / sigma2**2)
        g2 /= (g2.sum() + 1e-8)
        
        dog_kernel = (g1 - g2).view(1, 1, size, size, size)
        
        with torch.no_grad():
            conv.weight.copy_(dog_kernel.repeat(channels, 1, 1, 1, 1))
            conv.weight.requires_grad = True

    def forward(self, x1, x2):
        B, C, D, H, W = x1.shape
        y = x2 if x2 is not None else None

        assert B == 1
        x_deep = x1.contiguous()
        N = D * H * W

        energy = torch.norm(x_deep, dim=1, keepdim=False).view(-1) 
        k = max(1, int(N * self.keep_ratio))
        topk_vals, topk_indices = torch.topk(energy, k)
        dynamic_threshold = topk_vals[-1] 
        
        hard_mask = (energy >= dynamic_threshold).float()
        soft_mask = torch.sigmoid((energy - dynamic_threshold) * 10.0) 
        mask = (hard_mask - soft_mask).detach() + soft_mask
        mask = mask.view(1, 1, D, H, W)
        
        x_flat = x_deep.view(C, -1).t()  # [N, C]
        selected_points = torch.index_select(x_flat, 0, topk_indices)  # [M, C]
        
        if y is not None:
            y_dog = self.dog_conv(y).view(-1) # [N]
            selected_physical_prob = torch.index_select(y_dog, 0, topk_indices) # [M]
            
            d_min = selected_physical_prob.min()
            d_max = selected_physical_prob.max()
            selected_physical_prob = (selected_physical_prob - d_min) / (d_max - d_min + 1e-8) 
        else:
            selected_physical_prob = torch.ones_like(topk_vals)

        w_deep = F.softplus(self.gate_beta)
        deep_conf = self.sigmoid(w_deep * topk_vals)
        
        w_shallow = F.softplus(self.gate_dog)
        shallow_conf = self.sigmoid(w_shallow * selected_physical_prob)

        final_gate = shallow_conf * deep_conf

        cat_features = selected_points * final_gate.unsqueeze(-1)
        mamba_input = cat_features.unsqueeze(0).contiguous()
        x_norm = self.norm(mamba_input)
        mamba_out = self.mamba(x_norm.contiguous()).squeeze(0) # [M, C]
        
        mamba_out = mamba_out - mamba_out.mean(dim=0, keepdim=True)
        
        updates_flat = torch.zeros_like(x_flat)
        updates_flat[topk_indices] = mamba_out
        updates = updates_flat.t().view(1, C, D, H, W).contiguous()
        output = updates * mask 
        return output




############################## 推理使用 ######################################
# class RegMamba(nn.Module):
#     def __init__(self, dim, dim_shallow, d_state=16, d_conv=4, expand=2, keep_ratio=0.1): 
#         super().__init__()
#         self.dim = dim
#         self.keep_ratio = keep_ratio  
        
#         self.norm = nn.LayerNorm(dim)
#         self.mamba = Mamba(
#             d_model=dim,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#             use_fast_path=False
#         )
        
#         self.dog_conv = nn.Conv3d(dim_shallow, dim_shallow, kernel_size=5, 
#                                   padding=5//2, groups=dim_shallow, bias=False)
#         self._init_dog_weights(self.dog_conv, dim_shallow, sigma1=0.8, sigma2=1.6)

#         self.gate_beta = nn.Parameter(torch.zeros(1))  
#         self.gate_dog = nn.Parameter(torch.zeros(1))   

#         self.sigmoid = nn.Sigmoid()

#     def _init_dog_weights(self, conv, channels, sigma1, sigma2):
#         size = 5
#         ax = torch.linspace(-(size-1)/2., (size-1)/2., size)
#         zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing='ij')
#         dist_sq = xx**2 + yy**2 + zz**2
        
#         g1 = torch.exp(-0.5 * dist_sq / sigma1**2)
#         g1 /= (g1.sum() + 1e-8)
#         g2 = torch.exp(-0.5 * dist_sq / sigma2**2)
#         g2 /= (g2.sum() + 1e-8)
        
#         dog_kernel = (g1 - g2).view(1, 1, size, size, size)
        
#         with torch.no_grad():
#             conv.weight.copy_(dog_kernel.repeat(channels, 1, 1, 1, 1))
#             conv.weight.requires_grad = True

#     def forward(self, x1, x2):

#         B, C, D, H, W = x1.shape
#         N = D * H * W
        
#         x_deep = x1.permute(0, 2, 3, 4, 1).contiguous().view(B, N, C) 
        
#         energy = torch.norm(x_deep, dim=-1) # [B, N]
#         k = max(1, int(N * self.keep_ratio))
        
#         topk_vals, topk_indices = torch.topk(energy, k, dim=-1) # [B, k], [B, k]
        

#         idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, C) # [B, k, C]
#         selected_points = torch.gather(x_deep, 1, idx_expanded) # [B, k, C]

#         if x2 is not None:
#             y_dog = self.dog_conv(x2).view(B, -1) # [B, N]
#             selected_physical_prob = torch.gather(y_dog, 1, topk_indices) # [B, k]
            
#             d_min = selected_physical_prob.min(dim=-1, keepdim=True)[0] # [B, 1]
#             d_max = selected_physical_prob.max(dim=-1, keepdim=True)[0] # [B, 1]
#             selected_physical_prob = (selected_physical_prob - d_min) / (d_max - d_min + 1e-8) # [B, k]
#         else:
#             selected_physical_prob = torch.ones_like(topk_vals)


#         w_deep = F.softplus(self.gate_beta)
#         deep_conf = self.sigmoid(w_deep * topk_vals) # [B, k]
        
#         w_shallow = F.softplus(self.gate_dog)
#         shallow_conf = self.sigmoid(w_shallow * selected_physical_prob) # [B, k]

#         final_gate = (shallow_conf * deep_conf).unsqueeze(-1) # [B, k, 1]

#         # 前置干预：输入缩放
#         cat_features = selected_points * final_gate # [B, k, C]


#         mamba_input = cat_features.contiguous()
#         x_norm = self.norm(mamba_input)
#         mamba_out = self.mamba(x_norm) # [B, k, C]
        
#         mamba_out = mamba_out - mamba_out.mean(dim=1, keepdim=True)
        
#         updates_flat = torch.zeros_like(x_deep) # [B, N, C]
#         updates_flat.scatter_(dim=1, index=idx_expanded, src=mamba_out)
        
#         #  [B, N, C] -> [B, D, H, W, C] -> [B, C, D, H, W]
#         updates = updates_flat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        
#         return updates


class PGE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, 1, kernel_size=1) 
        self.fusion_channels = 7 
        self.av = nn.Conv3d(self.fusion_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # 定义 Sobel 算子 (固定权重)
        self.register_buffer('sobel_x', torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float().reshape(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float().reshape(1, 1, 3, 3))

    def get_gradient_info(self, x):
        
        B, C, D, H, W = x.shape
        img_flat = torch.mean(x, dim=1).view(B * D, 1, H, W)

        gx = F.conv2d(img_flat, self.sobel_x, padding=1)
        gy = F.conv2d(img_flat, self.sobel_y, padding=1)
 
        mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
        
        
        mag_safe = mag + 1e-8
        vec_x = gx / mag_safe # Cos
        vec_y = gy / mag_safe # Sin
        
      
        div_x = gx[:, :, :, 2:] - gx[:, :, :, :-2] 
        div_y = gy[:, :, 2:, :] - gy[:, :, :-2, :] 
        # padding 回去
        div_x = F.pad(div_x, (1, 1, 0, 0))
        div_y = F.pad(div_y, (0, 0, 1, 1))
        divergence = -(div_x + div_y) 

        mag = mag.view(B, 1, D, H, W)
        vec_x = vec_x.view(B, 1, D, H, W)
        vec_y = vec_y.view(B, 1, D, H, W)
        divergence = divergence.view(B, 1, D, H, W)
        
        return mag, vec_x, vec_y, divergence

    def forward(self, x):
        
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
 
        x1 = self.conv1(x)
        
       
        with torch.no_grad():
            
            grad_mag, vec_x, vec_y, div = self.get_gradient_info(x)
        
        concat = torch.cat([avg_pool, max_pool, x1, grad_mag, vec_x, vec_y, div], dim=1)
        
        tran = self.av(concat)
        attn_map = self.sigmoid(tran)
        
        return attn_map * x



class G_SAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.ehance = PGE(c1)
        self.mamba_layers = nn.ModuleList([
            RegMamba(dim=c2, dim_shallow=1, d_state=16, d_conv=4, expand=2) 
            for j in range(1)
        ])
        
        self.mlp_channel = nn.Sequential(
            LayerNorm(c2, data_format="channels_first"),
            MlpChannel(hidden_size=c2, mlp_dim=c2 * 4)
        )

        self.scale = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x1 = self.ehance(x) + x
        x2 = x1
        for layer in self.mamba_layers:
            x2 = layer(x2, y) 
        tran = x2 +  self.sigmoid(self.scale) * x1
        x3 = self.mlp_channel(tran)
        final = tran + x3
        return final
