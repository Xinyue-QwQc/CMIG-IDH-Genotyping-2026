"""
CTNet-3D: CNN-Transformer Hybrid Network for 3D Multimodal Classification

基于论文方法复现并适配3D体积数据:
- 输入: 体积数据 [B, C, D, H, W]
- 组件:
  1. 3D CNN Feature Extraction (替代EfficientNet/DenseNet)
  2. 3D Cross-Channel Attention Module (CCAM)
  3. 3D Swin Transformer
  4. Multimodal Fusion with Learnable Weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange
import math


class ChannelAttention3D(nn.Module):
    """
    3D Cross-Channel Attention Module (CCAM)
    
    结合Max Pooling和Average Pooling的通道注意力
    公式(8)(9): Att_c = σ(M1 · ReLU(M0 · F_c))
    """
    def __init__(self, channels: int, reduction: int = 16, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Global Pooling (3D版本)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # MLP (公式3: r = C/16)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            out: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        
        # Average Pooling分支 (公式8)
        avg_out = self.avg_pool(x).view(B, C)  # [B, C]
        avg_out = self.mlp(avg_out)
        
        # Max Pooling分支
        max_out = self.max_pool(x).view(B, C)  # [B, C]
        max_out = self.mlp(max_out)
        
        # 融合两个分支并应用sigmoid (公式9)
        attn = self.sigmoid(avg_out + max_out)  # [B, C]
        
        # 扩展维度并与输入相乘
        attn = attn.view(B, C, 1, 1, 1)
        return x * attn


class DenseBlock3D(nn.Module):
    """3D Dense Block (类似DenseNet)"""
    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))
    
    def _make_layer(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class CNNBackbone3D(nn.Module):
    """
    3D CNN Backbone (替代EfficientNet/DenseNet)
    适配输入: [B, 1, 128, 128, 128]
    输出: [B, C, 8, 8, 8] (便于后续Swin Transformer处理)
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        # 初始卷积: 128 -> 64
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # Stage 1: 64 -> 32
        self.stage1 = nn.Sequential(
            DenseBlock3D(base_channels, growth_rate=32, num_layers=3),
            nn.BatchNorm3d(base_channels + 3 * 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels + 3 * 32, base_channels * 2, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )
        
        # Stage 2: 32 -> 16
        self.stage2 = nn.Sequential(
            DenseBlock3D(base_channels * 2, growth_rate=64, num_layers=4),
            nn.BatchNorm3d(base_channels * 2 + 4 * 64),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2 + 4 * 64, base_channels * 4, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )
        
        # Stage 3: 16 -> 8
        self.stage3 = nn.Sequential(
            DenseBlock3D(base_channels * 4, growth_rate=128, num_layers=6),
            nn.BatchNorm3d(base_channels * 4 + 6 * 128),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4 + 6 * 128, base_channels * 8, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )
        
        # Stage 4 (Bottleneck): 保持8x8x8
        self.stage4 = nn.Sequential(
            DenseBlock3D(base_channels * 8, growth_rate=128, num_layers=3),
            nn.BatchNorm3d(base_channels * 8 + 3 * 128),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 8 + 3 * 128, base_channels * 8, kernel_size=1, bias=False),
        )
        
        # Channel Attention (CCAM)
        self.ccam = ChannelAttention3D(base_channels * 8, reduction=16)
        
        self.out_channels = base_channels * 8
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            feat: [B, C', D', H', W']
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.ccam(x)  # 应用通道注意力
        return x


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-head Self-Attention
    用于Swin Transformer 3D版本
    """
    def __init__(self, dim: int, num_heads: int = 8, window_size: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """
        将3D特征划分为窗口
        x: [B, D, H, W, C]
        """
        B, D, H, W, C = x.shape
        window_d, window_h, window_w = self.window_size
        
        # 确保尺寸可被窗口大小整除
        assert D % window_d == 0 and H % window_h == 0 and W % window_w == 0, \
            f"Spatial dimensions {D}x{H}x{W} must be divisible by window size {self.window_size}"
        
        # 重塑为窗口: [B, D//wd, wd, H//wh, wh, W//ww, ww, C]
        x = x.view(B, D // window_d, window_d, H // window_h, window_h, W // window_w, window_w, C)
        
        # 合并窗口和batch: [B * num_windows, wd * wh * ww, C]
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, window_d * window_h * window_w, C)
        
        return windows
    
    def window_reverse(self, windows: torch.Tensor, B: int, D: int, H: int, W: int) -> torch.Tensor:
        """恢复窗口到原始3D特征"""
        window_d, window_h, window_w = self.window_size
        
        # 从 [B * num_windows, wd * wh * ww, C] 恢复
        x = windows.view(B, D // window_d, H // window_h, W // window_w, 
                        window_d, window_h, window_w, self.dim)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, self.dim)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W, C]
        Returns:
            out: [B, D, H, W, C]
        """
        B, D, H, W, C = x.shape
        window_d, window_h, window_w = self.window_size
        
        # 如果空间尺寸小于窗口大小，使用全局注意力
        if D < window_d or H < window_h or W < window_w:
            # 全局自注意力
            x_flat = x.view(B, D * H * W, C)
            qkv = self.qkv(x_flat).reshape(B, D * H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            
            out = (attn @ v).transpose(1, 2).reshape(B, D * H * W, C)
            out = self.proj(out)
            out = out.view(B, D, H, W, C)
            return out
        
        # 划分窗口
        x_windows = self.window_partition(x)  # [B*nW, N, C]
        
        # QKV
        qkv = self.qkv(x_windows).reshape(x_windows.shape[0], x_windows.shape[1], 3, 
                                          self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(x_windows.shape[0], x_windows.shape[1], C)
        out = self.proj(out)
        
        # 恢复窗口
        out = self.window_reverse(out, B, D, H, W)
        
        return out


class SwinTransformerBlock3D(nn.Module):
    """
    3D Swin Transformer Block
    包含Window Attention和Shifted Window Attention
    """
    def __init__(self, dim: int, num_heads: int = 8, 
                 window_size: Tuple[int, int, int] = (4, 4, 4),
                 shift_size: Tuple[int, int, int] = (0, 0, 0),
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, window_size)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            out: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        shortcut = x
        
        # 转置为 [B, D, H, W, C] 用于LayerNorm
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        x = self.norm1(x)
        
        # Shifted Window (如果需要)
        if sum(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), 
                                  dims=(1, 2, 3))
        else:
            shifted_x = x
        
        # Window Attention
        attn_out = self.attn(shifted_x)
        
        # 反向shift
        if sum(self.shift_size) > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), 
                                 dims=(1, 2, 3))
        
        # 残差连接1
        x = shortcut + attn_out.permute(0, 4, 1, 2, 3)  # 转回 [B, C, D, H, W]
        
        # MLP
        shortcut2 = x
        x_norm = self.norm2(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = shortcut2 + self.mlp(x_norm.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        
        return x


class SwinTransformer3D(nn.Module):
    """
    3D Swin Transformer
    适配输入: [B, C, 8, 8, 8]
    PatchEmbed: stride=4 -> [B, embed_dim, 2, 2, 2]
    使用 window_size=(2,2,2) 适配小尺寸特征
    """
    def __init__(self, in_channels: int = 512, embed_dim: int = 96, 
                 num_heads: int = 8, window_size: Tuple[int, int, int] = (2, 2, 2),
                 shift_size: Tuple[int, int, int] = (1, 1, 1),
                 depths: list = [2, 2]):
        super().__init__()
        
        # Patch Embedding: 8x8x8 -> 2x2x2 (stride=4)
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=4, stride=4)
        
        # Swin Transformer Stages - 减少stage数量避免尺寸过小
        self.stages = nn.ModuleList()
        current_dim = embed_dim
        
        for i, depth in enumerate(depths):
            stage = nn.ModuleList()
            for j in range(depth):
                # 交替使用regular和shifted window
                shift = shift_size if j % 2 == 1 else (0, 0, 0)
                stage.append(
                    SwinTransformerBlock3D(
                        dim=current_dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=shift
                    )
                )
            self.stages.append(stage)
            
            # Patch Merging (除了最后一个stage)
            if i < len(depths) - 1:
                # 2x2x2 -> 1x1x1，使用stride=2下采样
                self.stages.append(
                    nn.Conv3d(current_dim, current_dim * 2, kernel_size=2, stride=2)
                )
                current_dim = current_dim * 2
        
        self.embed_dim = current_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            out: [B, C', D', H', W']
        """
        # Patch Embedding
        x = self.patch_embed(x)
        
        # Swin Transformer Stages
        for stage in self.stages:
            if isinstance(stage, nn.ModuleList):
                for block in stage:
                    x = block(x)
            else:
                x = stage(x)
        
        return x


class CTNet3D(nn.Module):
    """
    CTNet-3D: 完整的3D多模态分类网络
    适配输入: [B, 1, 128, 128, 128] x 2
    
    架构流程:
    1. CNN Feature Extraction -> [B, 512, 8, 8, 8]
    2. Cross-Channel Attention (CCAM)
    3. Swin Transformer Refinement -> [B, C, 1, 1, 1]
    4. Multimodal Fusion (公式5: F_fused = αF_h ⊙ βF_m)
    5. Classification
    """
    def __init__(self, 
                 in_ch_modality_a: int = 1,   # 例如CT
                 in_ch_modality_b: int = 1,   # 例如MRI
                 num_classes: int = 2,
                 base_channels: int = 64,
                 embed_dim: int = 96,
                 dropout: float = 0.5):
        super().__init__()
        
        # 模态A的特征提取
        self.cnn_a = CNNBackbone3D(in_ch_modality_a, base_channels)
        self.transformer_a = SwinTransformer3D(
            in_channels=self.cnn_a.out_channels,
            embed_dim=embed_dim
        )
        
        # 模态B的特征提取
        self.cnn_b = CNNBackbone3D(in_ch_modality_b, base_channels)
        self.transformer_b = SwinTransformer3D(
            in_channels=self.cnn_b.out_channels,
            embed_dim=embed_dim
        )
        
        # 可学习的融合权重 (公式5)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        # 全局池化 (Transformer输出可能已经是1x1x1，但保留以防万一)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Batch Normalization
        self.bn = nn.BatchNorm1d(self.transformer_a.embed_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer_a.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x) -> dict:
        """
        Args:
            x_a: 模态A输入 [B, C_a, D, H, W]
            x_b: 模态B输入 [B, C_b, D, H, W]
        Returns:
            dict with predictions and features
        """
        x_a,x_b = x[:,:2,...],x[:,2:,...]
        # 1. CNN特征提取 + CCAM (公式1, 7, 8, 9)
        feat_a = self.cnn_a(x_a)  # [B, C, D', H', W']
        feat_b = self.cnn_b(x_b)
        
        # 2. Swin Transformer精炼 (公式4)
        feat_a = self.transformer_a(feat_a)  # [B, C'', D'', H'', W'']
        feat_b = self.transformer_b(feat_b)
        
        # 3. 多模态融合 (公式5: F_fused = αF_h ⊙ βF_m)
        # 确保尺寸匹配
        if feat_a.shape != feat_b.shape:
            # 使用插值对齐
            feat_b = F.interpolate(feat_b, size=feat_a.shape[2:], mode='trilinear', align_corners=False)
        
        # 可学习权重的元素级乘法融合
        feat_fused = self.alpha * feat_a * self.beta * feat_b  # ⊙表示元素级乘法
        
        # 4. 全局池化 + BN (公式6)
        gap = self.global_pool(feat_fused).flatten(1)  # [B, C]
        gap = self.bn(gap)

        return gap 
        
        # 5. 分类 (公式10: Y_pred = softmax(W·F_fused + b))
        logits = self.classifier(gap)
        
        return logits
        # return {
        #     'logits': logits,
        #     'features': gap,
        #     'feat_a': feat_a,
        #     'feat_b': feat_b,
        #     'feat_fused': feat_fused
        # }


class FocalLoss(nn.Module):
    """
    Focal Loss for Class Imbalance (公式11)
    L = -α(1 - p_t)^γ log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============ 测试 ============
if __name__ == "__main__":
    device = torch.device( "cpu")
    
    # 创建CTNet-3D模型
    model = CTNet3D(
        in_ch_modality_a=2,   # CT
        in_ch_modality_b=2,   # MRI或其他模态
        num_classes=2,
        base_channels=64,     # 标准通道数
        embed_dim=96,
        dropout=0.5
    ).to(device)
    
    # 测试输入: [B, 1, 128, 128, 128]
    batch_size = 4
    x_a = torch.randn(batch_size, 4, 128, 128, 128).to(device)
    # x_b = torch.randn(batch_size, 1, 128, 128, 128).to(device)
    target = torch.randint(0, 2, (batch_size,)).to(device)
    
    print("=" * 60)
    print("CTNet-3D 测试")
    print(f"输入: 模态A={x_a.shape}")
    
    # 前向传播
    outputs = model(x_a)
    
    print("\n输出:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # 测试损失
    # criterion = FocalLoss(alpha=0.25, gamma=2.0)
    # loss = criterion(outputs['logits'], target)
    # print(f"\nFocal Loss: {loss.item():.4f}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("CTNet-3D 复现完成！")
    print("核心组件:")
    print("  1. 3D CNN + Cross-Channel Attention (CCAM)")
    print("  2. 3D Swin Transformer")
    print("  3. 可学习权重的多模态融合")
    print("  4. Focal Loss处理类别不平衡")
