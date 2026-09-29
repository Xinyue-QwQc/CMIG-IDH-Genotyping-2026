import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from utils.permute_operation import pr_permute, reverse_permute, permute
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
import time


# Come from repvit
class Conv3d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv3d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm3d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv3d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class LayerNorm3d(nn.LayerNorm):
    """ Layernorm for channels of '3d' spatial BCHWD tensors """

    def __init__(self, num_channels):
        super().__init__([num_channels, 1, 1, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class SE_Layer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed, flag=True) -> None:
        super().__init__()

        self.conv = Conv3d_BN(ed, ed, 5 if flag else 3, 1, 2 if flag else 1, groups=ed)
        self.conv1 = nn.Conv3d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = nn.BatchNorm3d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm3d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            out_dim = dim
        else:
            out_dim = dim * 2
        self.conv = Conv3d_BN(dim, out_dim, 3, 2, 1, groups=dim)

    def forward(self, x):
        return self.conv(x)


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W, D)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W, D = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(-1, window_size * window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, D):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W, D)
    """
    B = int(windows.shape[0] / (H * W * D / window_size / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size,
                        -1)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, -1, H, W, D)
    return x


class CNN_Block(nn.Module):
    def __init__(self, inp, bn_size, flag=True, use_se=True, growth_rate=24, drop_rate=0):
        super().__init__()
        oup = bn_size * growth_rate
        self.token_mixer = nn.Sequential(
            RepVGGDW(inp, flag),
            SE_Layer(inp, 4) if use_se else nn.Identity()
        )
        self.channel_mixer = nn.Sequential(
            Conv3d_BN(inp, oup, 1, 1, 0),
            nn.GELU(),
            Conv3d_BN(oup, growth_rate, 1, 1, 0, bn_weight_init=0)
        )

        self.drop_rate = drop_rate

    def forward(self, x):
        new_feature = self.channel_mixer(self.token_mixer(x))
        if self.drop_rate > 0:
            new_feature = F.dropout(new_feature,
                                    p=self.drop_rate,
                                    training=self.training)

        return torch.cat([x, new_feature], dim=1)


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


# 对Chanel进行划分，减少参数量的同时，让不同的channel学习到不同的信息，并进行交互
class MambaChannelMixerLayer(nn.Module):
    def __init__(self, d_model,
                 d_state=16,
                 reduction_ratio=16,
                 d_conv=4,
                 expand=2,
                 dt_rank="auto",
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 conv_bias=True,
                 bias=False,
                 use_fast_path=True,
                 layer_idx=None,
                 device=None,
                 dtype=None, ):
        super().__init__()
        """
        从三个轴面方向进行建模,然后再加权concat
        """

        self.d_model = d_model
        self.d_channel_model = d_model // 6

        # MambaVision原始的则是 d_state = 8, d_conv = 3, expand = 1
        self.mamba_channel_mixer_layer = nn.ModuleList(
            [
                MambaVisionMixer(
                    d_model=self.d_channel_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(6)
            ]
        )

        self.channel_mixer_weighted = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // reduction_ratio),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.d_model // reduction_ratio, self.d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, D = x.shape

        assert D % 6 == 0

        x_c = torch.mean(x, dim=1)
        x_c = self.channel_mixer_weighted(x_c)
        x_c = x_c.unsqueeze(1)

        x1, x2, x3, x4, x5, x6 = torch.chunk(x, 6, dim=-1)

        x1 = self.mamba_channel_mixer_layer[0](x1)

        # x2进行逆序
        x2 = x2.flip(1)
        x2 = self.mamba_channel_mixer_layer[1](x2)
        x2 = x2.flip(1)

        # x3进行permute 2,0,1  这部分代码有问题
        x3, permute_index = pr_permute(x3, [2, 0, 1])
        x3 = self.mamba_channel_mixer_layer[2](x3)
        x3 = reverse_permute(x3, permute_index)

        # x4进行permute 2,0,1 reverse
        x4 = permute(x4, permute_index)
        x4 = x4.flip(1)
        x4 = self.mamba_channel_mixer_layer[3](x4)
        x4 = x4.flip(1)
        x4 = reverse_permute(x4, permute_index)

        # x5进行permute 1 0 2
        x5, permute_index = pr_permute(x5, [1, 0, 2])
        x5 = self.mamba_channel_mixer_layer[4](x5)
        x5 = reverse_permute(x5, permute_index)

        # x6进行permute 1 0 2 reverse
        x6 = permute(x6, permute_index)
        x6 = x6.flip(1)
        x6 = self.mamba_channel_mixer_layer[5](x6)
        x6 = x6.flip(1)
        x6 = reverse_permute(x6, permute_index)

        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=-1)

        x = x * x_c

        return x


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=MLP,
                 layer_scale=None,
                 growth_rate=24,
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.is_attn = counter in transformer_blocks

        if self.is_attn:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaChannelMixerLayer(d_model=dim,
                                                d_state=8,
                                                d_conv=3,
                                                expand=2
                                                )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # if self.is_attn:
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                             act_layer=act_layer, drop=drop)
        # else:
        #     self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, out_features=growth_rate,
        #                             act_layer=act_layer, drop=drop)

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        # raw_features = x
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        # if self.is_attn:
        #     x = torch.cat([raw_features, x], dim=-1)
        return x


class MambaIDHLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 growth_rate=24
                 ):
        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([CNN_Block(inp=(dim + i * growth_rate),
                                                   bn_size=2, growth_rate=growth_rate, )
                                         for i in range(depth)])
            self.transformer_block = False
        else:
            self.transformer_block = True
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i,
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                         for i in range(depth)])

        if conv:
            new_dim = dim + growth_rate * depth
        else:
            new_dim = dim

        self.downsample = None if not downsample else Downsample(dim=new_dim, keep_dim=conv)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):

        _, _, H, W, D = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_g = (self.window_size - D % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0 or pad_g > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b, 0, pad_g))
                _, _, Hp, Wp, Dp = x.shape
            else:
                Hp, Wp, Dp = H, W, D
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp, Dp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W, :D].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)


class MambaIDH(nn.Module):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 growth_rate=24,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()

        new_dim = dim
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaIDHLayer(dim=new_dim,
                                  depth=depths[i],
                                  num_heads=num_heads[i],
                                  window_size=window_size[i],
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  qk_scale=qk_scale,
                                  conv=conv,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                  downsample=(i < 3),
                                  layer_scale=layer_scale,
                                  layer_scale_conv=layer_scale_conv,
                                  transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[
                                                                                                       i] % 2 != 0 else list(
                                      range(depths[i] // 2, depths[i])),
                                  )

            if conv:
                new_dim = new_dim + growth_rate * depths[i]
            else:
                new_dim = new_dim * 2

            self.levels.append(level)

        new_dim //= 2
        self.norm = nn.BatchNorm3d(new_dim)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(new_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def MambaIDH_T(num_classes, in_chans):
    model = MambaIDH(dim=96, in_chans=in_chans, in_dim=32, depths=[1, 3, 8, 4],
                     num_heads=[2, 4, 8, 16], window_size=[8, 8, 8, 4],
                     drop_path_rate=0.2, num_classes=num_classes, mlp_ratio=2)
    return model


if __name__ == "__main__":
    a = time.time()
    model = MambaIDH_T(2, 4).cuda()
    # print(model)
    t = torch.randn(4, 4, 128, 128, 128).cuda()
    print(model(t).shape)
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params / 1e6)
    b = time.time()
    print(b - a)