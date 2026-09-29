"""
HyPCA-Net 3D PyTorch Implementation (Memory Optimized Version)
修复了内存泄漏问题，优化了GPU内存使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


# ============================================================================
# 1. GlobalMinPooling3D - 全局最小池化层 (3D版本)
# ============================================================================
class GlobalMinPooling3D(nn.Module):
    """Global Minimum Pooling Layer for 3D inputs"""
    def __init__(self):
        super(GlobalMinPooling3D, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, C, D, H, W) -> Returns: (B, C)"""
        x = x.view(x.size(0), x.size(1), -1)  # (B, C, D*H*W)
        return torch.min(x, dim=2)[0]  # (B, C)


# ============================================================================
# 2. DeeperGlobalLocalAttentionLayer1 - 深度全局局部注意力层 (3D版本)
# 修复: 缓存DCT矩阵，避免重复创建
# ============================================================================
class DeeperGlobalLocalAttentionLayer1(nn.Module):
    """Multi-branch Fusion Attention (MFA) Module (3D版本) - Memory Optimized"""
    def __init__(self, units: int, activation: str = 'sigmoid', dropout_rate: float = 0.2,
                 use_scale: bool = True):
        super(DeeperGlobalLocalAttentionLayer1, self).__init__()
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_scale = use_scale
        
        # 使用 buffer 缓存 DCT 矩阵，避免每次 forward 重复创建
        self.register_buffer('_dct_matrix', None)
        self._dct_channels = None
        
        # 延迟初始化标志
        self._initialized = False
        self._channels = None

    def _build(self, channels: int):
        """构建网络层"""
        self.global_conv1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.global_conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.global_conv_3 = nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels)

        groups = min(8, channels)
        self.global_conv4 = nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=groups)

        self.global_attention = nn.Linear(channels, self.units)

        self.local_conv1 = nn.Conv3d(channels, channels, kernel_size=1, groups=channels)
        self.local_conv2 = nn.Conv3d(channels, channels, kernel_size=1, groups=channels)
        self.local_conv3 = nn.Conv3d(channels, channels, kernel_size=1, groups=channels)
        self.local_conv4 = nn.Conv3d(channels, channels, kernel_size=1, groups=channels)

        self.batch_norm = nn.LayerNorm(channels)

        if self.use_scale:
            self.global_scale = nn.Parameter(torch.ones(1))
            self.local_scale = nn.Parameter(torch.ones(1))

        self._initialized = True
        self._channels = channels
        self.global_min_pool = GlobalMinPooling3D()

    def _init_dct_matrix(self, C: int, dtype: torch.dtype, device: torch.device):
        """初始化并缓存 DCT 矩阵"""
        if self._dct_channels == C and self._dct_matrix is not None:
            return self._dct_matrix
        
        N = C
        n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)
        k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
        dct_matrix = torch.cos(math.pi / N * (n + 0.5) * k) * math.sqrt(2.0 / N)
        dct_matrix[0, :] *= 1.0 / math.sqrt(2)
        
        # 使用 register_buffer 确保矩阵随模型移动到正确的设备
        self._dct_matrix = dct_matrix
        self._dct_channels = C
        
        return dct_matrix

    def dct_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DCT transform along the channel dimension for 3D - Optimized"""
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)

        # 使用缓存的 DCT 矩阵
        dct_matrix = self._init_dct_matrix(C, x.dtype, x.device)

        dct_result = torch.matmul(x_flat, dct_matrix.T)
        dct_result = dct_result.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        return dct_result

    def get_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'gelu':
            return F.gelu(x)
        else:
            return torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape

        if not self._initialized:
            self._build(C)
            self.to(x.device)

        dct_transformed = self.dct_transform(x)

        global_attention1 = self.get_activation(self.global_conv1(dct_transformed))

        global_avg1 = F.adaptive_avg_pool3d(global_attention1, 1).view(B, C)
        global_max1 = F.adaptive_max_pool3d(global_attention1, 1).view(B, C)
        global_min1 = self.global_min_pool(global_attention1)

        dct_transformed_ga = x + global_attention1

        global_attention2 = self.get_activation(self.global_conv2(dct_transformed_ga))
        global_avg4 = F.adaptive_avg_pool3d(global_attention2, 1).view(B, C)
        global_max5 = F.adaptive_max_pool3d(global_attention2, 1).view(B, C)
        global_min6 = self.global_min_pool(global_attention2)

        dct_transformed_ga2 = dct_transformed_ga + global_attention2

        global_attention3 = self.get_activation(self.global_conv_3(dct_transformed_ga2))
        global_avg7 = F.adaptive_avg_pool3d(global_attention3, 1).view(B, C)
        global_max8 = F.adaptive_max_pool3d(global_attention3, 1).view(B, C)
        global_min9 = self.global_min_pool(global_attention3)

        global_concat_avg = global_avg1 + global_avg4 + global_avg7
        global_concat_max = global_max1 + global_max5 + global_max8
        global_concat_min = global_min1 + global_min6 + global_min9

        global_concat_add = global_concat_avg + global_concat_max + global_concat_min
        global_concat_sub = global_concat_max - global_concat_avg - global_concat_min

        global_avg_concat = global_concat_add + global_concat_sub
        global_attention = self.get_activation(self.global_attention(global_avg_concat))
        global_attention = global_attention.view(B, self.units, 1, 1, 1)

        # Local attention
        local_attention1 = self.get_activation(self.local_conv1(dct_transformed))
        local_attention1 = torch.mean(local_attention1, dim=[2, 3, 4], keepdim=True)

        local_attention2 = self.get_activation(self.local_conv2(dct_transformed))
        local_attention2 = torch.max(local_attention2.view(B, C, -1), dim=2, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)

        local_attention3 = self.get_activation(self.local_conv3(dct_transformed))
        local_attention3 = torch.min(local_attention3.view(B, C, -1), dim=2, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)

        # Matrix multiplication for local attention
        la1 = local_attention1.view(B, C, 1).permute(0, 2, 1)  # (B, 1, C)
        la2 = local_attention2.view(B, C, 1).permute(0, 2, 1)  # (B, 1, C)
        la3 = local_attention3.view(B, C, 1).permute(0, 2, 1)  # (B, 1, C)

        f = torch.matmul(la1, la2.transpose(-1, -2))  # (B, 1, 1)
        f_softmax = F.softmax(f, dim=-1)
        y = torch.matmul(f_softmax, la3)  # (B, 1, C)
        y = y.permute(0, 2, 1).view(B, C, 1, 1, 1)

        y = self.get_activation(self.local_conv4(y))
        y = y.permute(0, 2, 3, 4, 1)  # (B, 1, 1, 1, C) for LayerNorm
        y = self.batch_norm(y)
        y = y.permute(0, 4, 1, 2, 3)

        if self.use_scale:
            global_attention = global_attention * self.global_scale
            y = y * self.local_scale

        attention = torch.sigmoid(global_attention + y)

        return attention


# ============================================================================
# 3. MultiKernelGroupwiseConv2 - 多核分组卷积层 (3D版本)
# ============================================================================
class MultiKernelGroupwiseConv2(nn.Module):
    """Multi-Kernel Groupwise Convolution Block (3D版本)"""
    def __init__(self, filters: int, groups: int = 16, strides: int = 1):
        super(MultiKernelGroupwiseConv2, self).__init__()
        self.filters = filters
        self.groups = min(groups, filters)
        self.strides = strides

        self.conv1x1 = nn.Conv3d(filters, filters, kernel_size=1, stride=strides, padding=0, groups=filters)
        self.conv5x5 = nn.Conv3d(filters, filters, kernel_size=5, stride=strides, padding=2, groups=filters)
        self.conv7x7 = nn.Conv3d(filters, filters, kernel_size=7, stride=strides, padding=3, groups=filters)

        self.final_conv = nn.Conv3d(filters * 3, filters, kernel_size=1, groups=self.groups)
        self.shortcut_conv = nn.Conv3d(filters, filters, kernel_size=1, stride=strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1x1 = self.conv1x1(x)
        conv5x5 = self.conv5x5(x)
        conv7x7 = self.conv7x7(x)

        x1 = torch.cat([conv1x1, conv5x5, conv7x7], dim=1)
        x1 = self.final_conv(x1)

        shortcut = self.shortcut_conv(x)

        out = shortcut + x1
        out = F.relu(out)

        return out


# ============================================================================
# 4. ChannelSpatialAttention3D - 通道空间注意力 SCALA (3D版本)
# 修复: 提前初始化，避免延迟初始化问题
# ============================================================================
class ChannelSpatialAttention3D(nn.Module):
    """Spatial-Channel Adaptive Learning Attention (SCALA) (3D版本) - Memory Optimized"""
    def __init__(self, channels: int = 64):
        super(ChannelSpatialAttention3D, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        
        # 提前初始化，避免延迟初始化带来的问题
        self._channels = channels
        self.channel_conv = nn.Linear(channels, channels)
        self.spatial_conv = nn.Conv3d(1, 1, kernel_size=7, padding=3)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape

        # Channel attention
        avg_pool = torch.mean(x, dim=[2, 3, 4])
        max_pool = torch.max(x.view(B, C, -1), dim=2)[0]
        min_pool = torch.min(x.view(B, C, -1), dim=2)[0]
        sum_pool = torch.sum(x.view(B, C, -1), dim=2)

        pooled = avg_pool + max_pool + min_pool + sum_pool
        channel_info = torch.sigmoid(self.channel_conv(pooled))
        channel_info = channel_info.view(B, C, 1, 1, 1)

        # Spatial attention
        avg_pool_s = torch.mean(x, dim=1, keepdim=True)
        max_pool_s = torch.max(x, dim=1, keepdim=True)[0]
        min_pool_s = torch.min(x, dim=1, keepdim=True)[0]
        sum_pool_s = torch.sum(x, dim=1, keepdim=True)

        concat = avg_pool_s + max_pool_s + min_pool_s + sum_pool_s
        spatial_info = torch.sigmoid(self.spatial_conv(concat))

        attention_map = torch.sigmoid(channel_info + spatial_info)
        x_out = x * attention_map

        return x_out


# ============================================================================
# 5. MultiKernelGroupwiseConv - 多核分组卷积 (3D版本)
# 修复: 提前初始化 ChannelSpatialAttention3D
# ============================================================================
class MultiKernelGroupwiseConv3D(nn.Module):
    """Multi-Kernel Groupwise Convolution with optional SE attention (3D版本) - Memory Optimized"""
    def __init__(self, filters: int, groups: int = 4, strides: int = 1):
        super(MultiKernelGroupwiseConv3D, self).__init__()
        self.filters = filters
        self.groups = min(groups, filters)
        self.strides = strides

        self.conv1x1 = nn.Conv3d(filters, filters // 4, kernel_size=1, padding=0)
        self.dwconv3x3 = nn.Conv3d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.dwconv5x5 = nn.Conv3d(filters, filters, kernel_size=5, padding=2, groups=filters)
        self.dwconv_dilated = nn.Conv3d(filters, filters, kernel_size=3, padding=2, dilation=2, groups=filters)

        total_filters = filters * 3 + filters // 4
        self.post_shuffle_dwconv = nn.Conv3d(total_filters, total_filters, kernel_size=3, padding=1, groups=total_filters)
        self.post_shuffle_proj = nn.Conv3d(total_filters, filters, kernel_size=1, groups=self.groups)
        self.downsample_dwconv = nn.Conv3d(filters, filters, kernel_size=3, stride=strides, padding=1, groups=filters)

        self.shortcut_conv = nn.Conv3d(filters, filters, kernel_size=1, groups=self.groups)
        self.shortcut_dwconv = nn.Conv3d(filters, filters, kernel_size=3, stride=strides, padding=1, groups=filters)

        # 提前初始化，传入正确的通道数
        self.attention = ChannelSpatialAttention3D(channels=filters)

    def channel_shuffle(self, x: torch.Tensor, groups: int) -> torch.Tensor:
        B, C, D, H, W = x.shape
        g = min(groups, C)
        x = x.view(B, g, C // g, D, H, W)
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
        x = x.view(B, C, D, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1x1(x)
        conv3 = self.dwconv3x3(x)
        conv5 = self.dwconv5x5(x)
        conv_dilated = self.dwconv_dilated(x)

        x1 = torch.cat([conv1, conv3, conv5, conv_dilated], dim=1)
        x1 = self.channel_shuffle(x1, self.groups)

        x1 = self.post_shuffle_dwconv(x1)
        x1 = self.post_shuffle_proj(x1)
        x1 = self.downsample_dwconv(x1)

        x1 = self.attention(x1)

        x_shortcut = self.shortcut_conv(x)
        x_shortcut = self.shortcut_dwconv(x_shortcut)
        x_shortcut = self.shortcut_conv(x_shortcut)

        out = x_shortcut + x1
        out = F.relu(out)

        return out


# ============================================================================
# 6. AttentionBlock3D - 注意力块 MIFA (3D版本) - 修复版
# 修复: 缓存小波滤波器，避免重复创建
# ============================================================================
class AttentionBlock3D(nn.Module):
    """Multi-modal Interactive Fusion Attention (MIFA) Block (3D版本) - Memory Optimized"""
    def __init__(self, channels: int, reduction_ratio: int = 8, groups: int = 8,
                 strides: int = 1, use_scale: bool = True):
        super(AttentionBlock3D, self).__init__()
        self.channels = channels
        self.groups = min(groups, channels)
        self.strides = strides
        self.use_scale = use_scale

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.gmin = GlobalMinPooling3D()

        self.multi_kernel_groupwise_conv1 = MultiKernelGroupwiseConv3D(
            filters=channels, groups=self.groups, strides=self.strides
        )

        self.bn = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.25)

        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        self.weight4 = nn.Parameter(torch.ones(1))
        
        # 缓存小波滤波器
        self.register_buffer('_wavelet_lp_d', None)
        self.register_buffer('_wavelet_hp_d', None)
        self.register_buffer('_wavelet_lp_row', None)
        self.register_buffer('_wavelet_hp_row', None)
        self.register_buffer('_wavelet_lp_col', None)
        self.register_buffer('_wavelet_hp_col', None)
        self._wavelet_channels = None

    def _init_wavelet_filters(self, C: int, dtype: torch.dtype, device: torch.device):
        """初始化并缓存小波滤波器"""
        if self._wavelet_channels == C and self._wavelet_lp_d is not None:
            return
        
        sqrt2 = math.sqrt(2.0)

        lp_1d = torch.tensor([1.0/sqrt2, 1.0/sqrt2], dtype=dtype, device=device)
        hp_1d = torch.tensor([-1.0/sqrt2, 1.0/sqrt2], dtype=dtype, device=device)

        self._wavelet_lp_d = lp_1d.view(1, 1, 2, 1, 1).repeat(C, 1, 1, 1, 1)
        self._wavelet_hp_d = hp_1d.view(1, 1, 2, 1, 1).repeat(C, 1, 1, 1, 1)
        self._wavelet_lp_row = lp_1d.view(1, 1, 1, 1, 2).repeat(C, 1, 1, 1, 1)
        self._wavelet_hp_row = hp_1d.view(1, 1, 1, 1, 2).repeat(C, 1, 1, 1, 1)
        self._wavelet_lp_col = lp_1d.view(1, 1, 1, 2, 1).repeat(C, 1, 1, 1, 1)
        self._wavelet_hp_col = hp_1d.view(1, 1, 1, 2, 1).repeat(C, 1, 1, 1, 1)
        
        self._wavelet_channels = C

    def wavelet_dwt3d_conv(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Single-level 3D Haar Discrete Wavelet Transform via convolution - Optimized"""
        C = inputs.shape[1]
        
        # 确保滤波器已初始化
        self._init_wavelet_filters(C, inputs.dtype, inputs.device)

        low_d = F.conv3d(inputs, self._wavelet_lp_d, stride=(2, 1, 1), padding=(0, 0, 0), groups=C)
        high_d = F.conv3d(inputs, self._wavelet_hp_d, stride=(2, 1, 1), padding=(0, 0, 0), groups=C)

        low_d_low_row = F.conv3d(low_d, self._wavelet_lp_row, stride=(1, 1, 2), padding=(0, 0, 0), groups=C)
        low_d_high_row = F.conv3d(low_d, self._wavelet_hp_row, stride=(1, 1, 2), padding=(0, 0, 0), groups=C)
        high_d_low_row = F.conv3d(high_d, self._wavelet_lp_row, stride=(1, 1, 2), padding=(0, 0, 0), groups=C)
        high_d_high_row = F.conv3d(high_d, self._wavelet_hp_row, stride=(1, 1, 2), padding=(0, 0, 0), groups=C)

        cA = F.conv3d(low_d_low_row, self._wavelet_lp_col, stride=(1, 2, 1), padding=(0, 0, 0), groups=C)
        cH = F.conv3d(low_d_low_row, self._wavelet_hp_col, stride=(1, 2, 1), padding=(0, 0, 0), groups=C)
        cV = F.conv3d(low_d_high_row, self._wavelet_lp_col, stride=(1, 2, 1), padding=(0, 0, 0), groups=C)
        cD = F.conv3d(low_d_high_row, self._wavelet_hp_col, stride=(1, 2, 1), padding=(0, 0, 0), groups=C)
        cE = F.conv3d(high_d_low_row, self._wavelet_lp_col, stride=(1, 2, 1), padding=(0, 0, 0), groups=C)
        cF = F.conv3d(high_d_low_row, self._wavelet_hp_col, stride=(1, 2, 1), padding=(0, 0, 0), groups=C)
        cG = F.conv3d(high_d_high_row, self._wavelet_lp_col, stride=(1, 2, 1), padding=(0, 0, 0), groups=C)
        cH2 = F.conv3d(high_d_high_row, self._wavelet_hp_col, stride=(1, 2, 1), padding=(0, 0, 0), groups=C)

        return cA, cH, cV, cD, cE, cF, cG, cH2

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs1, inputs2 = inputs
        B, C, D, H, W = inputs1.shape

        # 3D Wavelet transform for inputs1
        coeffs = self.wavelet_dwt3d_conv(inputs1)

        gap_list = [torch.mean(c, dim=[2, 3, 4]) for c in coeffs]
        gmp_list = [torch.max(c.view(B, C, -1), dim=2)[0] for c in coeffs]
        gmin_list = [torch.min(c.view(B, C, -1), dim=2)[0] for c in coeffs]
        gsum_list = [torch.sum(c.view(B, C, -1), dim=2) for c in coeffs]

        gap_out = sum(gap_list)
        gmp_out = sum(gmp_list)
        gmin_out = sum(gmin_list)
        gsum_out = sum(gsum_list)

        combined_pool1 = gap_out + gmp_out + gmin_out + gsum_out
        combined_pool2 = gmp_out - (gap_out + gmin_out)

        # 应用 BN 和 reshape，确保输出是 (B, C, 1, 1, 1)
        combined_pool1 = self.bn(combined_pool1).view(B, C, 1, 1, 1)
        combined_pool2 = self.bn(combined_pool2).view(B, C, 1, 1, 1)

        combined_pool1 = combined_pool1 * self.weight1
        combined_pool2 = combined_pool2 * self.weight2

        combined_pool = combined_pool1 + combined_pool2

        # Process inputs2
        channel_attn = self.multi_kernel_groupwise_conv1(inputs2)

        gap_out2 = torch.mean(channel_attn, dim=[2, 3, 4])
        gmp_out2 = torch.max(channel_attn.view(B, C, -1), dim=2)[0]
        gmin_out2 = torch.min(channel_attn.view(B, C, -1), dim=2)[0]
        gsum_out2 = torch.sum(channel_attn.view(B, C, -1), dim=2)

        combined_pool3 = gap_out2 + gmp_out2 + gmin_out2 + gsum_out2
        combined_pool3 = self.bn(combined_pool3).view(B, C, 1, 1, 1)

        # Cross attention - 现在所有 tensor 都是 (B, C, 1, 1, 1)，可以直接相加
        combined_pool_weighted = (combined_pool * self.weight1) + (combined_pool3 * self.weight2)
        combined_pool_weighted = self.bn(combined_pool_weighted.view(B, C)).view(B, C, 1, 1, 1)
        combined_pool_weighted = self.dropout(combined_pool_weighted)
        combined_pool_weighted = combined_pool_weighted * self.weight1

        cross_attention1 = torch.sigmoid((combined_pool * self.weight1) + (combined_pool3 * self.weight2))
        cross_attention2 = torch.sigmoid(combined_pool_weighted)

        att1 = inputs1 * cross_attention1 * self.weight3
        att2 = inputs2 * cross_attention2 * self.weight4

        return att1, att2


# ============================================================================
# 7. NODEAttentionLayer3D - 神经ODE注意力层 (3D版本)
# 修复: 使用梯度检查点减少内存占用
# ============================================================================
class NODEAttentionLayer3D(nn.Module):
    """Neural ODE-based Attention Layer (3D版本) - Memory Optimized"""
    def __init__(self, dim: int, S_max: int = 4, se_ratio: int = 16, use_scale: bool = True,
                 use_gradient_checkpoint: bool = True):
        super(NODEAttentionLayer3D, self).__init__()
        self.dim = dim
        self.S_max = S_max
        self.use_scale = use_scale
        self.use_gradient_checkpoint = use_gradient_checkpoint

        bottleneck = max(1, dim // 4)

        self.mlp = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim),
            nn.Sigmoid()
        )

        self.time_head = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
            nn.Sigmoid()
        )

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.step_head = nn.Linear(dim, S_max)

        reduced_dim = max(1, dim // se_ratio)
        self.se_reduce = nn.Linear(dim, reduced_dim)
        self.se_expand = nn.Linear(reduced_dim, dim)

        if self.use_scale:
            self.global_scale = nn.Parameter(torch.ones(1))
            self.local_scale = nn.Parameter(torch.ones(1))

    def euler_step(self, h0: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        dh = self.mlp(h0)
        hT = h0 + dt.unsqueeze(-1) * dh
        return hT

    def rk2_step(self, h0: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        k1 = self.mlp(h0)
        mid = h0 + (dt.unsqueeze(-1) / 2) * k1
        k2 = self.mlp(mid)
        return h0 + dt.unsqueeze(-1) * k2

    def _ode_steps(self, h0: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """执行 ODE 步骤，可选择使用梯度检查点"""
        hT = h0
        for _ in range(self.S_max):
            h_e = self.euler_step(hT, dt)
            h_r = self.rk2_step(hT, dt)
            hT = h_r + h_e
        return hT

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        B, N, C = h0.shape

        h_mean = torch.mean(h0, dim=1)

        logits = self.step_head(h_mean)
        probs = F.softmax(logits, dim=-1)
        steps = torch.sum(probs * torch.arange(1, self.S_max + 1, dtype=probs.dtype, device=probs.device),
                         dim=-1, keepdim=True)

        dt = self.time_head(h_mean)

        # 使用梯度检查点减少内存占用
        if self.use_gradient_checkpoint and self.training:
            hT = torch.utils.checkpoint.checkpoint(
                self._ode_steps, h0, dt, use_reentrant=False
            )
        else:
            hT = self._ode_steps(h0, dt)

        gate_vals = self.gate(h0)
        out = gate_vals * hT + (1.0 - gate_vals) * h0

        se_mean = torch.mean(out, dim=1)
        se_max = torch.max(out, dim=1)[0]

        se = se_mean + se_max
        se = F.relu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        se = se.view(B, 1, C)

        out = out * se

        return out


# ============================================================================
# 8. MSWShiftedWindowDynamicFusion - 多尺度窗口动态融合 (3D版本)
# 修复: 优化窗口操作，减少临时张量
# ============================================================================
class MSWShiftedWindowDynamicFusion3D(nn.Module):
    """Multi-Scale Window Shifted Window Dynamic Fusion (3D版本) - Memory Optimized"""
    def __init__(self, dim: int, window_sizes: List[int] = [4, 8]):
        super(MSWShiftedWindowDynamicFusion3D, self).__init__()
        self.dim = dim
        self.window_sizes = window_sizes

        self.attn_layers = NODEAttentionLayer3D(dim)
        self.norm_layers = nn.LayerNorm(dim)

        self.freq_proj = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(len(window_sizes))
        ])

        self.sf_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            ) for _ in range(len(window_sizes))
        ])

        self.alpha = nn.Parameter(torch.ones(len(window_sizes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        scale_outputs = []
        scale_energies = []

        for i, win in enumerate(self.window_sizes):
            shift = win // 4

            # Shift
            x_shift = torch.roll(x, shifts=(shift, shift, shift), dims=(2, 3, 4))

            # Pad
            pad_d = (win - D % win) % win
            pad_h = (win - H % win) % win
            pad_w = (win - W % win) % win
            x_pad = F.pad(x_shift, (0, pad_w, 0, pad_h, 0, pad_d))
            Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

            # Partition into windows - 使用 reshape 代替 view 以提高安全性
            nD, nH, nW = Dp // win, Hp // win, Wp // win
            x_win = x_pad.reshape(B, C, nD, win, nH, win, nW, win)
            x_win = x_win.permute(0, 2, 4, 6, 3, 5, 7, 1)

            # Flatten windows
            x_win_flat = x_win.reshape(B * nD * nH * nW, win * win * win, C)

            # Energy
            energy = torch.mean(x_win_flat ** 2, dim=[1, 2])
            scale_energies.append(energy.mean())

            # Frequency projection
            fp = torch.sigmoid(self.freq_proj[i](x_win_flat))
            gate = self.sf_gates[i](x_win_flat)
            fused = gate * fp + (1.0 - gate) * x_win_flat

            # Attention
            attn_out = self.attn_layers(fused)

            # Reshape back
            attn_resh = attn_out.reshape(B, nD, nH, nW, win, win, win, C)
            attn_resh = attn_resh.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C, Dp, Hp, Wp)
            attn_unpad = attn_resh[:, :, :D, :H, :W]
            attn_final = torch.roll(attn_unpad, shifts=(-shift, -shift, -shift), dims=(2, 3, 4))

            # Add & norm
            scale_out = self.norm_layers((attn_final + x).permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            scale_outputs.append(scale_out)

        # Fuse across scales
        energies_tensor = torch.stack(scale_energies)
        e_w = F.softmax(energies_tensor, dim=0)
        a_w = F.softmax(self.alpha, dim=0)
        w = a_w * e_w
        w = w / (torch.sum(w) + 1e-8)

        out = sum(w[i] * scale_outputs[i] for i in range(len(self.window_sizes)))

        enhanced = x + out

        return enhanced


# ============================================================================
# 9. RGSA - 残差分组自注意力 (3D版本)
# ============================================================================
class RGSA3D(nn.Module):
    """Residual Group Self-Attention Block (3D版本)"""
    def __init__(self, filters: int, use_projection: bool = False):
        super(RGSA3D, self).__init__()
        self.filters = filters
        self.use_projection = use_projection

        self.conv1 = nn.Conv3d(filters, filters, kernel_size=1, groups=min(8, filters))
        self.dwconv1 = nn.Conv3d(filters, filters, kernel_size=5, padding=2, groups=filters)
        self.dwconv2 = nn.Conv3d(filters, filters, kernel_size=7, padding=3, groups=filters)

        self.bn1 = nn.BatchNorm3d(filters)
        self.bn2 = nn.BatchNorm3d(filters)

        self.conv2 = nn.Conv3d(filters, filters, kernel_size=1, groups=min(8, filters))
        self.dwconv3 = nn.Conv3d(filters, filters, kernel_size=5, padding=2, groups=filters)
        self.dwconv4 = nn.Conv3d(filters, filters, kernel_size=7, padding=3, groups=filters)

        self.bn3 = nn.BatchNorm3d(filters)

        if use_projection:
            self.shortcut_conv = nn.Conv3d(filters, filters, kernel_size=1)
            self.shortcut_bn = nn.BatchNorm3d(filters)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x1 = self.conv1(x)
        x2 = self.dwconv1(x)
        x3 = self.dwconv2(x)

        x = x1 + x2 + x3
        x = self.bn1(x)
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.dwconv3(x)
        x3 = self.dwconv4(x)

        x = x1 + x2 + x3
        x = self.bn2(x)
        x = F.relu(x)

        x = self.bn3(x)

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_bn(self.shortcut_conv(shortcut))

        x = x + shortcut
        x = F.relu(x)

        return x


# ============================================================================
# 10. HyPCA-Net3D 主网络架构 (3D版本) - 内存优化版
# ============================================================================
class HyPCANet3D(nn.Module):
    """HyPCA-Net3D: Hybrid Principal Component Analysis Network for 3D Medical Images - Memory Optimized"""
    def __init__(self, input_shape: Tuple[int, int, int, int] = (64, 64, 64, 1),
                 num_classes_1: int = 5,
                 dropout_rate: float = 0.25):
        super(HyPCANet3D, self).__init__()
        self.input_shape = input_shape
        self.num_classes_1 = num_classes_1
        self.dropout_rate = dropout_rate

        in_channels = input_shape[-1] if len(input_shape) == 4 else input_shape[0]

        # Initial convolution layers
        self.conv1_branch1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1_branch1 = nn.BatchNorm3d(64)
        self.pool1_branch1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.conv1_branch2 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1_branch2 = nn.BatchNorm3d(64)
        self.pool1_branch2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Stage 1: 64 channels
        self.rgsa1_branch1 = RGSA3D(64)
        self.rgsa1_branch2 = RGSA3D(64)
        self.attention1 = AttentionBlock3D(64)

        # Stage 2: 64 channels
        self.rgsa2_branch1 = RGSA3D(64)
        self.rgsa2_branch2 = RGSA3D(64)
        self.msw1_branch1 = MSWShiftedWindowDynamicFusion3D(64, window_sizes=[4, 8])
        self.msw1_branch2 = MSWShiftedWindowDynamicFusion3D(64, window_sizes=[4, 8])
        self.attention2 = AttentionBlock3D(64)

        # Stage 3: 128 channels
        self.transition_conv1 = nn.Conv3d(64, 128, kernel_size=1)
        self.transition_conv2 = nn.Conv3d(64, 128, kernel_size=1)
        self.rgsa3_branch1 = RGSA3D(128)
        self.rgsa3_branch2 = RGSA3D(128)
        self.attention3 = AttentionBlock3D(128)
        self.rgsa4_branch1 = RGSA3D(128)
        self.rgsa4_branch2 = RGSA3D(128)
        self.msw2_branch1 = MSWShiftedWindowDynamicFusion3D(128, window_sizes=[4, 8])
        self.msw2_branch2 = MSWShiftedWindowDynamicFusion3D(128, window_sizes=[4, 8])
        self.attention4 = AttentionBlock3D(128)

        # Stage 4: 256 channels
        self.transition_conv3 = nn.Conv3d(128, 256, kernel_size=1)
        self.transition_conv4 = nn.Conv3d(128, 256, kernel_size=1)
        self.rgsa5_branch1 = RGSA3D(256)
        self.rgsa5_branch2 = RGSA3D(256)
        self.attention5 = AttentionBlock3D(256)
        self.rgsa6_branch1 = RGSA3D(256)
        self.rgsa6_branch2 = RGSA3D(256)
        self.msw3_branch1 = MSWShiftedWindowDynamicFusion3D(256, window_sizes=[4, 8])
        self.msw3_branch2 = MSWShiftedWindowDynamicFusion3D(256, window_sizes=[4, 8])
        self.attention6 = AttentionBlock3D(256)

        # Stage 5: 512 channels
        self.transition_conv5 = nn.Conv3d(256, 512, kernel_size=1)
        self.transition_conv6 = nn.Conv3d(256, 512, kernel_size=1)
        self.rgsa7_branch1 = RGSA3D(512)
        self.rgsa7_branch2 = RGSA3D(512)
        self.attention7 = AttentionBlock3D(512)
        self.rgsa8_branch1 = RGSA3D(512)
        self.rgsa8_branch2 = RGSA3D(512)
        self.msw4_branch1 = MSWShiftedWindowDynamicFusion3D(512, window_sizes=[4, 8])
        self.msw4_branch2 = MSWShiftedWindowDynamicFusion3D(512, window_sizes=[4, 8])
        self.attention8 = AttentionBlock3D(512)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Global pooling and classifiers
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Classification heads
        self.classifier1 = nn.Linear(512 * 2, num_classes_1)

    def forward(self, x) -> torch.Tensor:
        x1, x2 = x[:,:2,...], x[:,2:,...]
        # Initial convolution
        x1 = F.relu(self.bn1_branch1(self.conv1_branch1(x1)))
        x1 = self.pool1_branch1(x1)

        x2 = F.relu(self.bn1_branch2(self.conv1_branch2(x2)))
        x2 = self.pool1_branch2(x2)

        # Stage 1
        x1 = self.rgsa1_branch1(x1)
        x1 = self.dropout(x1)
        x2 = self.rgsa1_branch2(x2)
        x2 = self.dropout(x2)
        x1, x2 = self.attention1((x1, x2))

        # Stage 2
        x1 = self.rgsa2_branch1(x1)
        x1 = self.dropout(x1)
        x2 = self.rgsa2_branch2(x2)
        x2 = self.dropout(x2)
        x1 = self.msw1_branch1(x1)
        x2 = self.msw1_branch2(x2)
        x1, x2 = self.attention2((x1, x2))

        # Stage 3: 128 channels
        x1 = self.transition_conv1(x1)
        x2 = self.transition_conv2(x2)
        x1 = self.rgsa3_branch1(x1)
        x1 = self.dropout(x1)
        x2 = self.rgsa3_branch2(x2)
        x2 = self.dropout(x2)
        x1, x2 = self.attention3((x1, x2))
        x1 = self.rgsa4_branch1(x1)
        x1 = self.dropout(x1)
        x2 = self.rgsa4_branch2(x2)
        x2 = self.dropout(x2)
        x1 = self.msw2_branch1(x1)
        x2 = self.msw2_branch2(x2)
        x1, x2 = self.attention4((x1, x2))

        # Stage 4: 256 channels
        x1 = self.transition_conv3(x1)
        x2 = self.transition_conv4(x2)
        x1 = self.rgsa5_branch1(x1)
        x1 = self.dropout(x1)
        x2 = self.rgsa5_branch2(x2)
        x2 = self.dropout(x2)
        x1, x2 = self.attention5((x1, x2))
        x1 = self.rgsa6_branch1(x1)
        x1 = self.dropout(x1)
        x2 = self.rgsa6_branch2(x2)
        x2 = self.dropout(x2)
        x1 = self.msw3_branch1(x1)
        x2 = self.msw3_branch2(x2)
        x1, x2 = self.attention6((x1, x2))

        # Stage 5: 512 channels
        x1 = self.transition_conv5(x1)
        x2 = self.transition_conv6(x2)
        x1 = self.rgsa7_branch1(x1)
        x2 = self.rgsa7_branch2(x2)
        x1, x2 = self.attention7((x1, x2))
        x1 = self.rgsa8_branch1(x1)
        x2 = self.rgsa8_branch2(x2)
        x1 = self.msw4_branch1(x1)
        x2 = self.msw4_branch2(x2)
        x1, x2 = self.attention8((x1, x2))

        # Concatenate branches
        x = torch.cat([x1, x2], dim=1)
        x = self.dropout(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification heads
        out1 = self.classifier1(x)
        return out1


# ============================================================================
# 辅助函数和测试代码
# ============================================================================
def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage():
    """获取当前 GPU 内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
    return "CUDA not available"


def test_model_3d():
    """Test the 3D model with random inputs"""
    print("=" * 70)
    print("HyPCA-Net3D PyTorch Implementation Test (Memory Optimized Version)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = HyPCANet3D(
        input_shape=(64, 64, 64, 2),  # 使用较小的测试尺寸
        num_classes_1=2,
        dropout_rate=0.25
    ).to(device)

    print(f"\nModel: HyPCA-Net3D (Memory Optimized)")
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 64, 64, 64).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"\nInitial memory: {get_memory_usage()}")

    model.eval()
    with torch.no_grad():
        try:
            out1 = model(x)

            print(f"\nInput shape: {x.shape}")
            print(f"Output shape: {out1.shape}")

            if torch.cuda.is_available():
                print(f"\nPeak memory: {get_memory_usage()}")

            print("\n" + "=" * 70)
            print("All tests completed successfully!")
            print("Memory optimizations applied:")
            print("  - DCT matrix caching")
            print("  - Wavelet filter caching")
            print("  - Gradient checkpointing in ODE layers")
            print("  - Eager initialization of attention modules")
            print("=" * 70)

        except Exception as e:
            print(f"\n[ERROR] Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    return model


if __name__ == "__main__":
    model = test_model_3d()
