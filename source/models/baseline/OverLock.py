'''
3D版本的OverLoCK模型，适用于医学影像处理
基于原2D版本修改：https://arxiv.org/abs/2502.20087
'''
import torch
import timm
import torch.distributed
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
# from mmengine.runner import load_checkpoint
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_3tuple  # 改为3D元组处理
from timm.models.registry import register_model


def get_conv3d(in_channels, 
               out_channels, 
               kernel_size, 
               stride, 
               padding, 
               dilation, 
               groups, 
               bias,
               attempt_use_lk_impl=True):
    """获取3D卷积层，替代原2D卷积"""
    kernel_size = to_3tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
    else:
        padding = to_3tuple(padding)
    need_large_impl = (kernel_size[0] == kernel_size[1] == kernel_size[2] and 
                      kernel_size[0] > 5 and 
                      padding == (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2))

    # 对于3D暂时不使用iGEMM实现，专注于功能正确性
    return nn.Conv3d(in_channels, out_channels, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding, 
                     dilation=dilation, 
                     groups=groups, 
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    """获取3D批归一化层"""
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm3d(dim)


def fuse_bn(conv, bn):
    """融合3D卷积和批归一化层"""
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1, 1),  # 适配3D权重形状
            bn.bias + (conv_bias - bn.running_mean) * bn.weight / std)

def convert_dilated_to_nondilated(kernel, dilate_rate):
    """将3D膨胀卷积转换为等效的非膨胀卷积"""
    identity_kernel = torch.ones((1, 1, 1, 1, 1)).to(kernel.device)  # 3D identity核
    if kernel.size(1) == 1:
        # 深度卷积
        dilated = F.conv_transpose3d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        # 普通卷积或分组卷积
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose3d(kernel[:,i:i+1,:,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    """将膨胀卷积合并到大型3D卷积核中"""
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    pad = large_k // 2 - equivalent_kernel_size // 2
    # 3D padding: (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [pad]*6)
    return merged_kernel


def stem(in_chans=1, embed_dim=96):  # 医学影像通常是单通道，如CT
    """3D输入stem模块"""
    return nn.Sequential(
        nn.Conv3d(in_chans, embed_dim//2, kernel_size=3, stride=(2,2,2), padding=1, bias=False),  # 保持深度，缩减H和W
        nn.BatchNorm3d(embed_dim//2),
        nn.GELU(),
        nn.Conv3d(embed_dim//2, embed_dim//2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(embed_dim//2),
        nn.GELU(),
        nn.Conv3d(embed_dim//2, embed_dim, kernel_size=3, stride=(2,2,2), padding=1, bias=False),  # 同时缩减D, H, W
        nn.BatchNorm3d(embed_dim),
        nn.GELU(),
        nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(embed_dim)
    )


def downsample(in_dim, out_dim):
    """3D下采样模块"""
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm3d(out_dim),
    )        


class SEModule(nn.Module):
    """3D空间注意力模块"""
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 3D平均池化
            nn.Conv3d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv3d(inner_dim, dim, kernel_size=1),
            out_act(),
        )
        
    def forward(self, x):
        x = x * self.proj(x)
        return x


class LayerScale(nn.Module):
    """3D层缩放模块"""
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1, 1)*init_value,  # 适配3D
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv3d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

        
class LayerNorm3d(nn.LayerNorm):
    """3D层归一化"""
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    
    def forward(self, x):
        x = rearrange(x, 'b c d h w -> b d h w c')  # 3D维度调整
        x = super().forward(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x.contiguous()


class GRN(nn.Module):
    """3D全局响应归一化"""
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))  # 3D形状
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2, -3), keepdim=True)  # 3D归一化
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x
    


class DilatedReparamBlock(nn.Module):
    """3D膨胀重参数化模块"""
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv3d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        # 3D调整的核大小和膨胀率设置
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'): # 部署模式
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv3d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))
       

class CTXDownsample(nn.Module):
    """3D上下文下采样模块"""
    def __init__(self, dim, h_dim):
        super().__init__()
        
        self.x_proj = nn.Sequential(
            nn.Conv3d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(h_dim)
        )
        self.h_proj = nn.Sequential(
            nn.Conv3d(h_dim//4, h_dim//4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(h_dim//4)
        )

    def forward(self, x, ctx):
        x = self.x_proj(x)
        ctx = self.h_proj(ctx)
        return (x, ctx)


class ResDWConv(nn.Conv3d):
    '''3D深度卷积带残差连接'''
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
    
    def forward(self, x):
        x = x + super().forward(x)
        return x


class RepConvBlock(nn.Module):
    """3D重参数化卷积块"""
    def __init__(self, 
                 dim=64,
                 kernel_size=7,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm3d,  # 使用3D层归一化
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False):
        super().__init__()
        
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint
        
        mlp_dim = int(dim*mlp_ratio)
        
        self.dwconv = ResDWConv(dim, kernel_size=3)
    
        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm3d(dim),
            SEModule(dim),
            nn.Conv3d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv3d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )

        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        
    def forward_features(self, x):
        
        x = self.dwconv(x)
        
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        return x
    
    def forward(self, x):
        
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)
        
        return x


class DynamicConvBlock(nn.Module):
    """3D动态卷积块"""
    def __init__(self,
                 dim=64,
                 ctx_dim=32,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm3d,  # 3D层归一化
                 is_first=False,
                 is_last=False,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False,
                 **kwargs):
        
        super().__init__()
        
        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim*mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint

        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1)
            self.h_scale = LayerScale(ctx_dim, init_value=1)
        
        self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
        self.norm1 = norm_layer(out_dim)
        
        self.fusion = nn.Sequential(
            nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm3d(out_dim),
            nn.GELU(),
            nn.Conv3d(out_dim, dim, kernel_size=1),
            GRN(dim),
        )
        
        self.weight_query = nn.Sequential(
            nn.Conv3d(dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm3d(dim//2),
        )
         
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool3d(7),  # 3D自适应池化
            nn.Conv3d(ctx_dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm3d(dim//2),
        )
        
        self.weight_proj = nn.Conv2d(343, kernel_size**3 + smk_size**3, kernel_size=1)  # 7^3=343 modify 3d
        
        self.dyconv_proj = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(dim),
        )
        
        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm3d(dim),
        )
        
        self.se_layer = SEModule(dim)
        
        self.gate = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.SiLU(),
        )

        self.proj = nn.Sequential(
            nn.BatchNorm3d(dim),
            nn.Conv3d(dim, out_dim, kernel_size=1),
        )
        
        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)
        
        self.mlp = nn.Sequential(
            nn.Conv3d(out_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv3d(mlp_dim, out_dim, kernel_size=1),
        )
        
        self.ls1 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.get_rpb()


    def get_rpb(self):
        """获取3D相对位置偏置"""
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1, self.rpb_size1))  # 3D
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2, self.rpb_size2))  # 3D
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)
    
        
    @torch.no_grad()
    def generate_idx(self, kernel_size):
        """生成3D索引"""
        rpb_size = 2 * kernel_size - 1
        idx_d = torch.arange(0, kernel_size)
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_d.unsqueeze(-1).unsqueeze(-1) * rpb_size * rpb_size) + 
                 (idx_h.unsqueeze(-1) * rpb_size) + 
                 idx_w).view(-1)
        return (idx_d, idx_h, idx_w, idx_k)
    

    def apply_rpb(self, attn, rpb, depth, height, width, kernel_size, idx_d, idx_h, idx_w, idx_k):
        """应用3D相对位置偏置"""
        rpb_size = 2 * kernel_size - 1
        num_repeat_d = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        
        num_repeat_d[kernel_size//2] = depth - (kernel_size-1)
        num_repeat_h[kernel_size//2] = height - (kernel_size-1)
        num_repeat_w[kernel_size//2] = width - (kernel_size-1)
        
        bias_dhw = ((idx_d.repeat_interleave(num_repeat_d).unsqueeze(-1).unsqueeze(-1) * rpb_size * rpb_size) +
                   (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * rpb_size) +
                   idx_w.repeat_interleave(num_repeat_w))
        
        bias_idx = bias_dhw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size**3))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 3)[:, bias_idx]  # 3D flatten
        rpb = rpb.reshape(1, int(self.num_heads), int(depth), int(height), int(width), int(kernel_size**3))
        return attn + rpb
    

    def _forward_inner(self, x, h_x, h_r):
        input_resoltion = x.shape[2:]  # (D, H, W)
        B, C, D, H, W = x.shape
        B, C_h, D_h, H_h, W_h = h_x.shape
        
        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_f = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f
        x_f = self.norm1(x_f)
        x = self.fusion(x_f)
        gate = self.gate(x)
        lepe = self.lepe(x)

        is_pad = False
        if min(D, H, W) < self.kernel_size:
            is_pad = True
            # 保持比例的3D插值
            scale = self.kernel_size / min(D, H, W)
            size = (int(D * scale), int(H * scale), int(W * scale))
            x = F.interpolate(x, size=size, mode='trilinear', align_corners=False)
            x_f = F.interpolate(x_f, size=size, mode='trilinear', align_corners=False)
            D, H, W = size

        query, key = torch.split(x_f, split_size_or_sections=[C, C_h], dim=1)
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)
        query = rearrange(query, 'b (g c) d h w -> b g c (d h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) d h w -> b g c (d h w)', g=self.num_heads)
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (d h w) -> b g d h w l', d=D, h=H, w=W)

        # 分割3D核权重
        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size**3, self.kernel_size**3], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, D, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, D, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(x, 'b (m g c) d h w -> m b g d h w c', m=2, g=self.num_heads)

        # 3D动态卷积应用
        x1 = F.conv3d(value[0].flatten(0, 1), 
                      attn1.permute(0, 1, 5, 2, 3, 4).flatten(0, 1),
                      groups=self.num_heads * B,
                      padding=self.smk_size//2)
        x1 = rearrange(x1, '(b g) c d h w -> b (g c) d h w', b=B, g=self.num_heads)
        
        x2 = F.conv3d(value[1].flatten(0, 1), 
                      attn2.permute(0, 1, 5, 2, 3, 4).flatten(0, 1),
                      groups=self.num_heads * B,
                      padding=self.kernel_size//2)
        x2 = rearrange(x2, '(b g) c d h w -> b (g c) d h w', b=B, g=self.num_heads)

        x = torch.cat([x1, x2], dim=1)

        if is_pad:
            x = F.interpolate(x, size=input_resoltion, mode='trilinear', align_corners=False)

        x = self.dyconv_proj(x)

        x = x + lepe
        x = self.se_layer(x)

        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))
        
        x = self.dwconv2(x)
         
        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if self.is_last:
            return (x, None)
        else:
            l_x, h_x = torch.split(x, split_size_or_sections=[C, C_h], dim=1)
            return (l_x, h_x)
    
    def forward(self, x, h_x, h_r):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
        else:
            x = self._forward_inner(x, h_x, h_r)
        return x


class OverLoCK3D(nn.Module):
    '''
    3D版本的OverLoCK模型，适用于医学影像
    基于原2D版本修改: https://arxiv.org/abs/2502.20087
    '''
    def __init__(self, 
                 depth=[2, 2, 2, 2],
                 sub_depth=[4, 2],
                 in_chans=4,  # 医学影像通常为单通道
                 embed_dim=[96, 192, 384, 768],
                 kernel_size=[7, 7, 7, 7],
                 mlp_ratio=[4, 4, 4, 4],
                 sub_mlp_ratio=[4, 4],
                 sub_num_heads=[4, 8],
                 ls_init_value=[None, None, 1, 1],
                 res_scale=True,
                 smk_size=3,  # 3D模型使用较小的核以减少计算量
                 deploy=False,
                 use_gemm=False,
                 use_ds=True,
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_layer=LayerNorm3d,  # 3D层归一化
                 projection=1024,
                 num_classes=2,  # 医学影像任务通常是二分类或少数类别
                 use_checkpoint=[0, 0, 0, 0],
            ):
 
        super().__init__()
        
        fusion_dim = embed_dim[-1] + embed_dim[-1]//4
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed1 = stem(in_chans, embed_dim[0])
        self.patch_embed2 = downsample(embed_dim[0], embed_dim[1])
        self.patch_embed3 = downsample(embed_dim[1], embed_dim[2])
        self.patch_embed4 = downsample(embed_dim[2], embed_dim[3])
        self.high_level_proj = nn.Conv3d(embed_dim[-1], embed_dim[-1]//4, kernel_size=1)
        self.patch_embedx = CTXDownsample(embed_dim[2], embed_dim[3])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth) + sum(sub_depth))]

        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.blocks3 = nn.ModuleList()
        self.blocks4 = nn.ModuleList()
        self.sub_blocks3 = nn.ModuleList()
        self.sub_blocks4 = nn.ModuleList()
        
        for i in range(depth[0]):
            self.blocks1.append(
                RepConvBlock(
                    dim=embed_dim[0],
                    kernel_size=kernel_size[0],
                    mlp_ratio=mlp_ratio[0],
                    ls_init_value=ls_init_value[0],
                    res_scale=res_scale,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[0]),
                )
            )
        
        for i in range(depth[1]):
            self.blocks2.append(
                RepConvBlock(
                    dim=embed_dim[1],
                    kernel_size=kernel_size[1],
                    mlp_ratio=mlp_ratio[1],
                    ls_init_value=ls_init_value[1],
                    res_scale=res_scale,
                    drop_path=dpr[i+depth[0]],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[1]),
                )
            )
            
        for i in range(depth[2]):
            self.blocks3.append(
                RepConvBlock(
                    dim=embed_dim[2],
                    kernel_size=kernel_size[2],
                    mlp_ratio=mlp_ratio[2],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth[:2])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[2]),
                )
            )

        for i in range(depth[3]):
            self.blocks4.append(
                RepConvBlock(
                    dim=embed_dim[3],
                    kernel_size=kernel_size[3],
                    mlp_ratio=mlp_ratio[3],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth[:3])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[3]),
                )
            )
            
        for i in range(sub_depth[0]):
            self.sub_blocks3.append(
                DynamicConvBlock(
                    dim=embed_dim[2],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[2],
                    num_heads=sub_num_heads[0],
                    mlp_ratio=sub_mlp_ratio[0],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth)],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    is_first=(i==0),
                    use_checkpoint=(i<use_checkpoint[2]),
                )
            )
        
        for i in range(sub_depth[1]):
            self.sub_blocks4.append(
                DynamicConvBlock(
                    dim=embed_dim[3],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[-1],
                    num_heads=sub_num_heads[1],
                    mlp_ratio=sub_mlp_ratio[1],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth)+sub_depth[0]],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    is_first=False,
                    is_last=(i==sub_depth[1]-1),
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[3]),
                )
            )

        # 辅助分类头
        if use_ds:
            self.aux_head = nn.Sequential(
                nn.BatchNorm3d(embed_dim[-1]),
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(embed_dim[-1], num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
            )
        
        # 主分类头
        self.head = nn.Sequential(
            nn.Conv3d(fusion_dim, projection, kernel_size=1, bias=False),
            nn.BatchNorm3d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
        )
        
        self.apply(self._init_weights)
        
        if torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def reparam(self):
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()
            
    def forward_pre_features(self, x):
        """前向传播预处理特征"""
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
            
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        return x
    
    
    def forward_base_features(self, x):
        """前向传播基础特征"""
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
            
        ctx = self.patch_embed4(x)
        for blk in self.blocks4:
            ctx = blk(ctx)

        return (x, ctx)
    

    def forward_sub_features(self, x, ctx):
        """前向传播子特征"""
        ctx_cls = ctx
        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)

        x, ctx = self.patch_embedx(x, ctx)
        for idx, blk in enumerate(self.sub_blocks4):
            x, ctx = blk(x, ctx, ctx_ori)
        
        return (x, ctx_cls)

    def forward_features(self, x):
        """前向传播特征提取"""
        x = self.forward_pre_features(x)
        x, ctx = self.forward_base_features(x)
        x, ctx_cls = self.forward_sub_features(x, ctx)

        return (x, ctx_cls)

    def forward(self, x):
        """前向传播"""
        x, ctx = self.forward_features(x)
        x = self.head(x).flatten(1)

        if hasattr(self, 'aux_head') and self.training:
            ctx = self.aux_head(ctx).flatten(1)
            return dict(main=x, aux=ctx)
        
        return x


# 模型配置
def _cfg3d(url=None, **kwargs):
    return {
        'url': url,
        'num_classes': 2,  # 医学影像通常是二分类
        'input_size': (4, 128, 128, 128),  # 3D医学影像输入尺寸 (C, D, H, W)
        'mean': [0.5],  # 单通道均值
        'std': [0.5],   # 单通道标准差
        'classifier': 'head',** kwargs,
    }


# 3D模型实例
@register_model
def overlock3d_t(pretrained=False, pretrained_cfg=None, **kwargs):
    """小型3D OverLoCK模型"""
    model = OverLoCK3D(
        depth=[2, 2, 3, 2],
        sub_depth=[4, 2],
        embed_dim=[48, 96, 192, 384],  # 减小维度以适应3D计算
        kernel_size=[5, 5, 5, 3],  # 3D使用较小的核
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[2, 4],
        sub_mlp_ratio=[3, 3],** kwargs
    )
    
    model.default_cfg = _cfg3d()
    return model


@register_model
def overlock3d_s(pretrained=False, pretrained_cfg=None, **kwargs):
    """中型3D OverLoCK模型"""
    model = OverLoCK3D(
        depth=[3, 3, 6, 3],
        sub_depth=[6, 3],
        embed_dim=[64, 128, 256, 512],
        kernel_size=[7, 7, 5, 5],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[4, 8],
        sub_mlp_ratio=[3, 3],** kwargs
    )
    
    model.default_cfg = _cfg3d()
    return model


if __name__ == '__main__':
    # 测试3D模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = overlock3d_t(num_classes=2).to(device)
    model.eval()

    # 医学影像输入 (B, C, D, H, W)
    x = torch.randn(1, 1, 128, 128, 128).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"输出形状: {y.shape}")  # 应为 (1, 2)