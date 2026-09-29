"""
ImprovedMamba3D - 3D版本的多尺度Mamba网络
适配输入: [B, C, D, H, W] - 例如医学影像体积数据
"""
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as F
from einops import rearrange
import typing as t



class SCSA3D(nn.Module):
    """3D版本的SCSA (Spatial-Channel Self Attention)"""
    def __init__(self,
                 dim: int,
                 head_num: int,
                 window_size: int = 7,
                 group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
                 qkv_bias: bool = False,
                 attn_drop_ratio: float = 0.,
                 gate_layer: str = 'sigmoid',
                 ):
        super(SCSA3D, self).__init__()

        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.dim // 4

        # 3D版本的深度卷积（深度可分离卷积）
        self.local_dwc = nn.Conv3d(self.group_chans, self.group_chans, 
                                   kernel_size=(1, self.group_kernel_sizes[0], self.group_kernel_sizes[0]),
                                   padding=(0, self.group_kernel_sizes[0] // 2, self.group_kernel_sizes[0] // 2), 
                                   groups=self.group_chans)
        self.global_dwcs = nn.ModuleList([
            nn.Conv3d(self.group_chans, self.group_chans, 
                      kernel_size=(1, size, size), 
                      padding=(0, size // 2, size // 2), 
                      groups=self.group_chans)
            for size in self.group_kernel_sizes[1:]
        ])

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.norm_d = nn.GroupNorm(4, dim)
        self.norm = nn.GroupNorm(1, dim)

        # 查询、键、值卷积层 (3D)
        self.q = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        # 下采样函数
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.down_func = nn.AvgPool3d(kernel_size=(1, window_size, window_size), 
                                          stride=(1, window_size, window_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d_, h_, w_ = x.size()

        # 计算三个方向的特征 (depth, height, width)
        x_h = x.mean(dim=4)  # [B, C, D, H]
        x_w = x.mean(dim=3)  # [B, C, D, W]
        x_d = x.mean(dim=(3, 4))  # [B, C, D]

        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # 计算水平注意力 (沿着H维度)
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h.unsqueeze(4)).squeeze(4),
            *[gw(l_x_h.unsqueeze(4)).squeeze(4) for gw in self.global_dwcs]
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, d_, h_, 1)

        # 计算垂直注意力 (沿着W维度)
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w.unsqueeze(3)).squeeze(3),
            *[gw(l_x_w.unsqueeze(3)).squeeze(3) for gw in self.global_dwcs]
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, d_, 1, w_)

        # 计算深度注意力 (沿着D维度)
        x_d_expanded = x_d.unsqueeze(3).unsqueeze(4)  # [B, C, D, 1, 1]
        l_x_d = x_d_expanded[:, :self.group_chans, ...]
        x_d_attn = self.sa_gate(self.norm_d(torch.cat((
            self.local_dwc(l_x_d),
            *[gw(l_x_d) for gw in self.global_dwcs]
        ), dim=1)))
        x_d_attn = x_d_attn.view(b, c, d_, 1, 1)

        # 计算最终的加权结果
        x = x * x_h_attn * x_w_attn * x_d_attn

        # 通道自注意力
        y = self.down_func(x)
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        _, _, d_new, h_new, w_new = y.size()

        # 调整维度以进行注意力计算
        q = rearrange(q, 'b (head_num head_dim) d h w -> b head_num head_dim (d h w)', 
                      head_num=self.head_num, head_dim=self.head_dim)
        k = rearrange(k, 'b (head_num head_dim) d h w -> b head_num head_dim (d h w)', 
                      head_num=self.head_num, head_dim=self.head_dim)
        v = rearrange(v, 'b (head_num head_dim) d h w -> b head_num head_dim (d h w)', 
                      head_num=self.head_num, head_dim=self.head_dim)

        # 计算注意力
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))

        # 加权值
        attn = attn @ v
        attn = rearrange(attn, 'b head_num head_dim (d h w) -> b (head_num head_dim) d h w', 
                         d=d_new, h=h_new, w=w_new)

        # 计算通道注意力
        attn = attn.mean((2, 3, 4), keepdim=True)
        attn = self.ca_gate(attn)

        return attn * x


class PyramidAttention3D(nn.Module):
    """3D金字塔注意力"""
    def __init__(self, dim, num_heads, bias):
        super(PyramidAttention3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # QKV卷积 (3D)
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, 
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, d, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', 
                       head=self.num_heads, d=d, h=h, w=w)

        out = self.project_out(out)
        return out


class PyramidRefinedChannelAttention3D(nn.Module):
    """3D金字塔精炼通道注意力"""
    def __init__(self, dim, num_heads, bias, num_scales=3, num_layers=2):
        super(PyramidRefinedChannelAttention3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.attention_modules = nn.ModuleList([
            PyramidAttention3D(dim, num_heads, bias) for _ in range(num_scales)
        ])

        self.attention_layers = nn.ModuleList([
            nn.ModuleList([PyramidAttention3D(dim, num_heads, bias) for _ in range(num_layers)])
            for _ in range(num_scales)
        ])

        self.project_out = nn.Conv3d(dim * num_scales, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, d, h, w = x.shape
        outputs = []

        for i, attention_module in enumerate(self.attention_modules):
            if i == 0:
                scaled_input = x
            else:
                # 3D下采样
                scaled_input = F.avg_pool3d(x, kernel_size=(2 ** i, 2 ** i, 2 ** i), 
                                           stride=(2 ** i, 2 ** i, 2 ** i))

            output = attention_module(scaled_input)

            for layer in self.attention_layers[i]:
                output = layer(output)

            if i > 0:
                # 3D上采样
                output = F.interpolate(output, size=(d, h, w), mode='trilinear', align_corners=False)

            outputs.append(output)

        out = torch.cat(outputs, dim=1)
        out = self.project_out(out)
        return out


class MultiScaleConv3D(nn.Module):
    """3D多尺度卷积"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv3D, self).__init__()
        # 3D卷积核: (depth, height, width)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.final_conv = nn.Conv3d(out_channels * 4, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.ReLU()
        self.attention = ChannelAttention3D(out_channels)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.final_conv(out)
        out = self.norm(out)
        out = self.activation(out)
        return self.attention(out)


class ChannelAttention3D(nn.Module):
    """3D通道注意力"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return x * self.sigmoid(y)


class DynamicConvBlock3D(nn.Module):
    """3D动态卷积块"""
    def __init__(self, channels, kernel_size=3, num_experts=4, reduction=4, dropout=0.1):
        super(DynamicConvBlock3D, self).__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.kernel_size = kernel_size

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels // reduction, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.convs = nn.ModuleList([
            nn.Conv3d(channels, channels, kernel_size=(1, kernel_size, kernel_size), 
                     padding=(0, kernel_size // 2, kernel_size // 2), groups=channels)
            for _ in range(num_experts)
        ])

        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        B, C, D, H, W = x.shape

        attention_weights = self.attention(x).view(B, self.num_experts, 1, 1, 1, 1)

        outputs = [conv(x).unsqueeze(1) for conv in self.convs]
        outputs = torch.cat(outputs, dim=1)  # [B, num_experts, C, D, H, W]

        x = (outputs * attention_weights).sum(dim=1)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ImprovedSpeMamba3D(nn.Module):
    """3D光谱Mamba模块"""
    def __init__(self, channels, token_num=4, use_residual=True, group_num=4, num_scales=3, num_layers=2):
        super(ImprovedSpeMamba3D, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.pyramid_refined_attention = PyramidRefinedChannelAttention3D(
            dim=self.channel_num,
            num_heads=4,
            bias=True,
            num_scales=num_scales,
            num_layers=num_layers
        )

        # Mamba保持1D处理，但输入需要重新reshape
        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, D, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, D, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_re = self.pyramid_refined_attention(x_pad)

        B, C, D, H, W = x_re.shape
        # 3D reshape: 将D,H,W展平为一个维度给Mamba处理
        x_re_flat = x_re.view(B * D * H * W, self.token_num, self.group_channel_num)
        x_recon = self.mamba(x_re_flat)

        x_recon = x_recon.view(B, C, D, H, W)
        x_recon = self.proj(x_recon)

        return x_recon + x if self.use_residual else x_recon


class ImprovedSpaMamba3D(nn.Module):
    """3D空间Mamba模块"""
    def __init__(self, channels, use_residual=True, group_num=4, token_num=4, num_scales=3, num_layers=2):
        super(ImprovedSpaMamba3D, self).__init__()
        self.use_residual = use_residual
        self.token_num = token_num
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.pyramid_refined_attention = PyramidRefinedChannelAttention3D(
            dim=self.channel_num,
            num_heads=4,
            bias=True,
            num_scales=num_scales,
            num_layers=num_layers
        )

        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.multi_scale_conv = MultiScaleConv3D(channels, channels)
        self.scsa = SCSA3D(dim=channels, head_num=4, window_size=7)

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, x):
        x_re = self.pyramid_refined_attention(x)

        B, C, D, H, W = x_re.shape
        x_flat = x_re.view(B * D * H * W, 1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, C, D, H, W)
        x_recon = self.proj(x_recon)

        return x_recon + x if self.use_residual else x_recon


class ImprovedBothMamba3D(nn.Module):
    """3D双分支Mamba融合"""
    def __init__(self, channels, token_num, use_residual, group_num=4):
        super(ImprovedBothMamba3D, self).__init__()
        self.use_residual = use_residual

        self.spa_mamba = ImprovedSpaMamba3D(channels, use_residual=use_residual, group_num=group_num)
        self.spe_mamba = ImprovedSpeMamba3D(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

        self.attention = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)

        fused_input = torch.cat([spa_x, spe_x], dim=1)
        attention_map = self.attention(fused_input)

        spa_x_attended = spa_x * attention_map
        spe_x_attended = spe_x * (1 - attention_map)

        fusion_x = spa_x_attended + spe_x_attended

        return fusion_x + x if self.use_residual else fusion_x


class ImprovedMamba3D(nn.Module):
    """
    完整的3D Mamba网络
    适配体积数据: [B, C, D, H, W]
    """
    def __init__(self, in_channels=1, hidden_dim=64, num_classes=2, use_residual=True, mamba_type='both',
                 token_num=4, group_num=4):
        super(ImprovedMamba3D, self).__init__()
        self.mamba_type = mamba_type

        # 3D Patch Embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        # 3D Mamba模块
        if mamba_type == 'spa':
            self.mamba = nn.Sequential(
                ImprovedSpaMamba3D(hidden_dim, use_residual=use_residual, group_num=group_num),
                nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(
                ImprovedSpeMamba3D(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num),
                nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            )
        elif mamba_type == 'both':
            self.mamba = nn.Sequential(
                ImprovedBothMamba3D(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num),
                nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            )

        # 3D动态卷积
        self.dynamic_conv = DynamicConvBlock3D(channels=hidden_dim)

        # 3D分类头
        self.cls_head = nn.Sequential(
            nn.Conv3d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv3d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, D, H, W] - 3D体积数据
        Returns:
            logits: [B, num_classes, D/2, H/2, W/2]
        """
        x = self.patch_embedding(x)
        x = self.mamba(x)  # 下采样到 D/2, H/2, W/2
        x = self.dynamic_conv(x)
        logits = self.cls_head(x)
        logits = torch.mean(logits, dim=(2, 3, 4))
        return logits


# ============ 测试 ============
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试3D网络
    model = ImprovedMamba3D(
        in_channels=4,      # 例如单通道CT
        hidden_dim=64,
        num_classes=2,
        use_residual=True,
        mamba_type='both',  # 'spa', 'spe', or 'both'
        token_num=4,
        group_num=4
    ).to(device)
    
    # 模拟3D输入: [B, C, D, H, W]
    # batch_size = 2
    # x = torch.randn(batch_size, 4, 128,128,128).to(device)  # 小型体积用于测试
    
    print("=" * 60)
    print("ImprovedMamba3D 测试")
    # print(f"输入尺寸: {x.shape}")
    
    # 前向传播
    # logits = model(x)
    # print(f"输出尺寸: {logits.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params/1e6:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("3D版本适配完成！")
    print("输入: [B, C, D, H, W] - 体积数据")
    print("输出: [B, num_classes, D/2, H/2, W/2] - 分割/分类特征图")
