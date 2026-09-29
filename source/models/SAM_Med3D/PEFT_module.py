import torch.nn as nn 
import torch
import math
from typing import Optional
from einops import  rearrange
# from .model_util import GRN_3D


import torch
import torch.nn.functional as F
import torch.nn as nn


class GroupBatchnorm3d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm3d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W, D = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W, D)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm3d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv3d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv3d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv3d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv3d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv3d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        """
         op_channel: int,
         group_num: int = 4,
         gate_treshold: float = 0.5,
         alpha: float = 1 / 2,
         squeeze_radio: int = 2,
         group_size: int = 2,
         group_kernel_size: int = 3,
        """
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

# 原文是act之后再使用grn
class GRN_3D(nn.Module):
    def __init__(self,dim,mode="Last") -> None:
        super().__init__()
        if mode == "Last":
            self.gamma = nn.Parameter(torch.zeros(1,1,1,1,dim))
            self.beta = nn.Parameter(torch.zeros(1,1,1,1,dim))
        else:
            self.gamma = nn.Parameter(torch.zeros(1,dim,1,1,1))
            self.beta = nn.Parameter(torch.zeros(1,dim,1,1,1))

        # 标识channel的位置
        self.mode = mode
    
    def forward(self, x):
        if self.mode == "Last":
            gx = torch.norm(x,p=2,dim=(1,2,3),keepdim=True)
            nx = gx /(gx.mean(dim=-1,keepdim=True)+1e-6)
        else:
            gx = torch.norm(x,p=2,dim=(2,3,4),keepdim=True)
            nx = gx /(gx.mean(dim=1,keepdim=True)+1e-6)
        return self.gamma*(x*nx) + self.beta + x 

class Med_adapter(nn.Module):
    def __init__(self,in_channels,hidden_channels,input_size:Optional[tuple]=None):
        super().__init__()
        self.linear_down = nn.Linear(in_channels,hidden_channels)

        self.conv_3_3_3 = nn.Sequential(nn.Conv3d(hidden_channels,hidden_channels,groups=hidden_channels,kernel_size=(1,3,3),padding=(0,1,1)),
                                        nn.Conv3d(hidden_channels,hidden_channels,groups=hidden_channels,kernel_size=(3,1,1),padding=(1,0,0)))

        if input_size is None:
            h = w = d = 8  # Default size
        else:
            h, w, d = input_size
        # self.fft_w
        # self.fft_b
        self.new_d = d // 2 + 1

        self.conv_5_5_5 = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, groups=hidden_channels, kernel_size=(5,1,1),
                      padding=(2, 0, 0)),
            nn.Conv3d(hidden_channels, hidden_channels, groups=hidden_channels, kernel_size=(1,5,5),
                      padding=(0, 2,2))
        )

        self.mixer = nn.Conv3d(hidden_channels,hidden_channels,kernel_size=1)

        # self.complex_weight = torch.randn(hidden_channels, h, w, self.new_d, 2, dtype=torch.float32) * 0.02
        # radius_h, radius_w, radius_d = h//2, w//2, self.new_d//2

        # 初始化低通滤波核
        # self.complex_weight[:, radius_h-1: radius_h+1,
        # radius_w-1 : radius_w+1, radius_d-1 : radius_d, :] = 0

        self.complex_weight = nn.Parameter(torch.randn(hidden_channels, h, w, self.new_d, 2, dtype=torch.float32) * 0.02)

        # self.complex_weight = nn.Parameter(torch.randn(hidden_channels, h, w, self.new_d, 2, dtype=torch.float32)*0.02)
        self.complex_bias = nn.Parameter(torch.randn(hidden_channels, 2, dtype=torch.float32) * 0.02)

        self.up_sample = nn.Linear(hidden_channels, in_channels)

    def forward(self,x):
        residual = x
        x = self.linear_down(x)
        x = x.permute(0,4,1,2,3).contiguous()
        residual2 = x

        x_3 = self.conv_3_3_3(x)
        x_5 = self.conv_5_5_5(x)

        B,C,D,H,W = x.shape

        x_fre = torch.fft.rfftn(x,dim=(2,3,4), norm="ortho")

        weight = torch.view_as_complex(self.complex_weight)
        bias = torch.view_as_complex(self.complex_bias).reshape(1,-1,1,1,1)

        x_fre = x_fre*weight + bias

        x_fft = torch.fft.irfftn(x_fre, s = (H,W,D),norm="ortho")

        x_multi_scale = x_3 + x_5 + x_fft

        x_fused = self.mixer(x_multi_scale)
        x_fused += residual2

        x_fused = x_fused.permute(0,2,3,4,1).contiguous()

        x_up = self.up_sample(x_fused)
        return x_up + residual




# SElayer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class spatial_Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(2,1,kernel_size=3,padding=1,bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        # B C D H W
        x_max,_ = torch.max(x,dim=1,keepdim=True)
        x_mean = torch.mean(x,dim=1,keepdim=True)
        y = torch.cat([x_max,x_mean],dim=1)
        y = self.conv(y)
        return x*self.sigmoid(y)

class DepthWiseConv(nn.Module):

    def __init__(self, in_channel, out_channel):

        super(DepthWiseConv, self).__init__()

        # 逐通道卷积 groups控制分组卷积
        self.depth_conv = nn.Conv3d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv3d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads: int):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q.to(self.q_proj.weight.dtype))
        k = self.k_proj(k.to(self.k_proj.weight.dtype))
        v = self.v_proj(v.to(self.v_proj.weight.dtype))

        # q = self.q_proj(q)
        # k = self.k_proj(k)
        # v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class GITCrossAdapter_FFT(nn.Module):
    """
       进行图文融合对齐，并充当adapter的作用进行微调
       in_channels: 输入到模型中的微调参数
       reduction_ratio: 维度缩减的范围
       """
    def __init__(self, in_channels, text_in_channels=1024, reduction_ratio=16) -> None:
        super().__init__()
        hidden_state = in_channels // reduction_ratio

        self.norm = nn.LayerNorm(2 * hidden_state)
        self.down = nn.Linear(in_channels, hidden_state)
        self.act = nn.GELU()

        self.text_prj = nn.Linear(text_in_channels, 2 * hidden_state, bias=False)
        self.conv_dw = nn.Conv3d(hidden_state, hidden_state, kernel_size=3, stride=1, padding=1, groups=hidden_state)



        # self.complex_weight = nn.Parameter(torch.randn(hidden_state, 8, 8, 5, 2, dtype=torch.float32) * 0.02)
        # self.complex_bias = nn.Parameter(torch.randn(hidden_state, 2, dtype=torch.float32) * 0.02)
        self.se = SELayer(hidden_state, reduction=reduction_ratio)
        self.sp = spatial_Attention()

        self.up = nn.Linear(hidden_state, in_channels)

    def forward(self, img, text):
        text = self.text_prj(text)
        text = self.norm(text)
        scale, shift = text.chunk(2, dim=-1)

        raw_img = img
        img = self.down(img)
        img = self.act(img)

        # B, H, W, D, C = img.shape
        img = img.permute(0, 4, 1, 2, 3).contiguous()
        residual = img
        img = self.conv_dw(img)

        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        img = img * (1 + scale) + shift
        # Pre Residual
        # img += residual
        # CBAM Module
        # img = self.se(img)

        B, C, D, H, W = img.shape

        x_fre = torch.fft.rfftn(img, dim=(2, 3, 4), norm="ortho")

        weight = torch.view_as_complex(self.complex_weight)
        bias = torch.view_as_complex(self.complex_bias).reshape(1, -1, 1, 1, 1)

        x_fre = x_fre * weight + bias

        img = torch.fft.irfftn(x_fre, s=(H, W, D), norm="ortho")

        img += residual
        img = rearrange(img, "B C H W D -> B H W D C").contiguous()
        img = self.up(img)
        return img + raw_img

class GITCrossAblation(nn.Module):
    """
            进行图文融合对齐,并充当adapter的作用进行微调
            in_channels: 输入到模型中的微调参数
            reduction_ratio: 维度缩减的范围
            """

    def __init__(self, in_channels, text_in_channels=1024, reduction_ratio=16) -> None:
        super().__init__()
        hidden_state = in_channels // reduction_ratio

        self.norm = nn.LayerNorm( hidden_state)  # 2 *
        self.down = nn.Linear(in_channels, hidden_state)
        self.act = nn.GELU()

        self.text_prj = nn.Linear(text_in_channels, 2 * hidden_state, bias=False)  #
        self.conv_dw = nn.Conv3d(hidden_state, hidden_state, kernel_size=3, stride=1, padding=1, groups=hidden_state)

        self.scconv = ScConv(hidden_state)

        self.up = nn.Linear(hidden_state, in_channels)

    def forward(self, img, text):
        # 对文本进行处理
        text = self.text_prj(text)
        text = self.norm(text)
        scale, shift = text.chunk(2, dim=-1)

        # raw_img = img + raw_img
        img = self.down(img)
        img = self.act(img)

        # B, H, W, D, C = img.shape
        img = img.permute(0, 4, 1, 2, 3).contiguous()
        img = self.conv_dw(img)
        # text = text.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # img += text

        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        img = img * (1 + scale) + shift

        residual = img
        img = self.scconv(img)
        img += residual

        img = rearrange(img, "B C H W D -> B H W D C").contiguous()
        img = self.up(img)
        return img

class GITCrossAdapter(nn.Module):
    """
        进行图文融合对齐,并充当adapter的作用进行微调
        in_channels: 输入到模型中的微调参数
        reduction_ratio: 维度缩减的范围
        """

    def __init__(self, in_channels, text_in_channels=1024, reduction_ratio=16) -> None:
        super().__init__()
        hidden_state = in_channels // reduction_ratio

        self.norm = nn.LayerNorm(2 * hidden_state)  #
        self.down = nn.Linear(in_channels, hidden_state)
        self.act = nn.GELU()

        self.text_prj = nn.Linear(text_in_channels,  2 * hidden_state, bias=False)  #
        self.conv_dw = nn.Conv3d(hidden_state, hidden_state, kernel_size=3, stride=1, padding=1, groups=hidden_state)

        self.scconv = ScConv(hidden_state)

        self.up = nn.Linear(hidden_state, in_channels)

    def forward(self, img, text):
        # 对文本进行处理
        text = self.text_prj(text)
        text = self.norm(text)
        scale, shift = text.chunk(2, dim=-1)

        # raw_img = img + raw_img
        img = self.down(img)
        img = self.act(img)

        # B, H, W, D, C = img.shape
        img = img.permute(0, 4, 1, 2, 3).contiguous()
        img = self.conv_dw(img)
        # text = text.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # img += text

        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        img = img * (1 + scale) + shift

        residual = img
        img = self.scconv(img)
        img += residual

        img = rearrange(img, "B C H W D -> B H W D C").contiguous()
        img = self.up(img)
        return img



class BasicLayer(nn.Module):
    def __init__(self,conv_layer,dim,act_fn=nn.GELU(),norm_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            conv_layer,
            norm_layer(dim),
        )
        self.act = act_fn
    def forward(self,x):
        return self.act(self.layer(x))

# Resnet瓶颈段
class ExpertA(nn.Module):
    def __init__(self,in_channels,hidden_channels,act_layer=nn.LeakyReLU,bn_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        # 先降维
        self.layer0 = BasicLayer(nn.Conv3d(in_channels,hidden_channels,kernel_size=1,bias=False),
                                 hidden_channels)
        self.layer1 = BasicLayer(nn.Conv3d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False,groups=hidden_channels),
                                 hidden_channels)
        self.layer2 = BasicLayer(nn.Conv3d(hidden_channels,in_channels,kernel_size=1,bias=False),
                                 in_channels,act_fn=nn.Identity())
        # self.act_fn = act_layer(inplace=True)
        self.act_fn = act_layer()
        # self.conv0 = nn.Conv3d(in_channels,hidden_channels,kernel_size=1,bias=False)
        # self.bn0 = bn_layer(hidden_channels)
        # self.conv1 = nn.Conv3d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False)
        # self.bn1 = bn_layer(hidden_channels)
        # self.conv2 = nn.Conv3d(hidden_channels,in_channels,kernel_size=1,bias=False)
        # self.bn2 = bn_layer(in_channels)
        # self.act_fn = act_layer(inplace=True)

        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        )
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        residual = x
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + residual
        x = self.act_fn(x)
        return x

# 扩大感受野
class ExpertA_v2(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels=0,act_layer=nn.LeakyReLU,bn_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        if out_channels == 0:
            out_channels = in_channels
        self.layer0 = BasicLayer(nn.Conv3d(in_channels,hidden_channels,kernel_size=3,padding=1,bias=False),hidden_channels)
        self.layer1 = BasicLayer(nn.Conv3d(hidden_channels,hidden_channels,kernel_size=3,padding=2,dilation=2,bias=False),hidden_channels)
        self.layer2 = BasicLayer(nn.Conv3d(hidden_channels,out_channels,kernel_size=3,padding=3,dilation=3,bias=False),
                                 hidden_channels,act_fn=nn.Identity())
        self.act_fn = act_layer(inplace=True)

        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        )
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        residual = x
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + residual
        x = self.act_fn(x)
        return x



# 通道注意力机制
class ExpertB(nn.Module):
    def __init__(self,in_channels,hidden_channels,act_layer=nn.LeakyReLU,bn_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        # 先降维
        self.conv0 = nn.Conv3d(in_channels,hidden_channels,kernel_size=1,bias=False)
        self.bn0 = bn_layer(hidden_channels)
        self.conv1 = SELayer(hidden_channels)
        self.bn1 = bn_layer(hidden_channels)
        self.conv2 = nn.Conv3d(hidden_channels,in_channels,kernel_size=1,bias=False)
        self.bn2 = bn_layer(in_channels)
        # self.act_fn = act_layer(inplace=True)
        self.act_fn = act_layer()

        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        )
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        residual = x
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act_fn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.act_fn(x)
        return x


class ExpertB_v2(nn.Module):
    def __init__(self,in_channels,hidden_channels,act_layer=nn.LeakyReLU,bn_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        self.layer0 = BasicLayer(nn.Conv3d(in_channels,hidden_channels,kernel_size=3,padding=1,bias=False),hidden_channels)
        self.layer1 = BasicLayer(SELayer(hidden_channels,reduction=4),hidden_channels)
        self.layer2 = BasicLayer(nn.Conv3d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False),
                                 hidden_channels,act_fn=nn.Identity())
        self.act_fn = act_layer(inplace=True)

    def forward(self,x):
        residual = x
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + residual
        x = self.act_fn(x)
        return x
     
# 空间注意力机制
class ExpertC(nn.Module):
    def __init__(self,in_channels,hidden_channels,act_layer=nn.LeakyReLU,bn_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        # 先降维
        self.conv0 = nn.Conv3d(in_channels,hidden_channels,kernel_size=1,bias=False)
        self.bn0 = bn_layer(hidden_channels)
        self.conv1 = spatial_Attention()
        self.bn1 = bn_layer(hidden_channels)
        self.conv2 = nn.Conv3d(hidden_channels,in_channels,kernel_size=1,bias=False)
        self.bn2 = bn_layer(in_channels)
        # self.act_fn = act_layer(inplace=True)
        self.act_fn = act_layer()

        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        )
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        residual = x
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act_fn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.act_fn(x)
        return x

class ExpertC_v2(nn.Module):
    def __init__(self,in_channels,hidden_channels,act_layer=nn.LeakyReLU,bn_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        self.layer0 = BasicLayer(nn.Conv3d(in_channels,hidden_channels,kernel_size=3,padding=1,bias=False),hidden_channels)
        self.layer1 = BasicLayer(spatial_Attention(),hidden_channels)
        self.layer2 = BasicLayer(nn.Conv3d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False),
                                    hidden_channels,act_fn=nn.Identity())
        self.act_fn = act_layer(inplace=True)
    def forward(self,x):
        residual = x
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + residual
        x = self.act_fn(x)
        return x


# 深度可分离卷积
class ExpertD(nn.Module):
    def __init__(self,in_channels,hidden_channels,act_layer=nn.LeakyReLU,bn_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        # 先降维
        self.conv0 = nn.Conv3d(in_channels,hidden_channels,kernel_size=1,bias=False)
        self.bn0 = bn_layer(hidden_channels)
        self.conv1 = DepthWiseConv(hidden_channels,hidden_channels)
        self.bn1 = bn_layer(hidden_channels)
        self.conv2 = nn.Conv3d(hidden_channels,in_channels,kernel_size=1,bias=False)
        self.bn2 = bn_layer(in_channels)
        # self.act_fn = act_layer(inplace=True)
        self.act_fn = act_layer()

        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        )
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        residual = x
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act_fn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.act_fn(x)
        return x

class ExpertD_v2(nn.Module):
    def __init__(self,in_channels,hidden_channels,act_layer=nn.LeakyReLU,bn_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        self.layer0 = BasicLayer(nn.Conv3d(in_channels,hidden_channels,kernel_size=3,padding=1,bias=False),hidden_channels)
        self.layer1 = BasicLayer(DepthWiseConv(hidden_channels,hidden_channels),hidden_channels)
        self.layer2 = BasicLayer(nn.Conv3d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False),
                                    hidden_channels,act_fn=nn.Identity())
        self.act_fn = act_layer(inplace=True)
    def forward(self,x):
        residual = x
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + residual
        x = self.act_fn(x)
        return x

# 创建一个卷积adapter的moe，相当于是异质集成
class Conv_adapter_moe(nn.Module):
    def __init__(self,dim=768,hidden_dim=96,num_experts=4,split_num=64) -> None:
        super().__init__()
        self.gate = nn.Linear(dim,num_experts,bias=False)
        # self.noise = nn.Parameter(torch.randn(dim,num_experts))
        # self.pool = nn.AdaptiveAvgPool3d(1)
        self.soft = nn.Softmax(dim=-1)
        self.num_experts = num_experts

        self.pre_norm = nn.LayerNorm(dim)
        self.experts = nn.ModuleList()
        self.experts.append(ExpertA(dim,hidden_dim,act_layer=nn.GELU))
        self.experts.append(ExpertB(dim,hidden_dim,act_layer=nn.GELU))
        self.experts.append(ExpertC(dim,hidden_dim,act_layer=nn.GELU))
        self.experts.append(ExpertD(dim,hidden_dim,act_layer=nn.GELU))

        self.norm_layer = nn.LayerNorm(dim)

    
    def forward(self,x):
        
        B,D,H,W,C = x.size()
        # x_gate = x.permute(0,4,1,2,3).contiguous()
        # B,C
        # x_gate = self.pool(x_gate).reshape(B,-1)
        x = self.pre_norm(x)
        x_values = self.gate(x) 
        x_values = self.soft(x_values)
        # B 768 8 8 8
        x = x.permute(0,4,1,2,3).contiguous()
        
        # B 8 8 8 768 × 4 
        conv_x_ls = [expert(x).permute(0,2,3,4,1).contiguous() for expert in self.experts]
        # B 8 8 8 768 4 
        # conv_experts = torch.stack(conv_x_ls,dim=-1)

        results = x_values[...,0].unsqueeze(-1)* conv_x_ls[0]
        for i in range(1,self.num_experts):
            results += x_values[...,i].unsqueeze(-1)* conv_x_ls[i]
        
        results = self.norm_layer(results)
        return results



class Conv_adapter_moe_v2(nn.Module):
    def __init__(self,dim=768,num_experts=4,split_num=64) -> None:
        super().__init__()
        assert dim % split_num == 0
        cubedim = round(split_num **(1/3))
        assert  cubedim**3 == split_num
        self.cubedim = int(cubedim)
        # 通道数减少64倍，H W D 各增大4倍
        self.gate = nn.Linear(dim,num_experts,bias=False)
        self.soft = nn.Softmax(dim=-1)
        self.num_experts = num_experts

        splited_dim = dim//split_num
        self.experts = nn.ModuleList()
        self.experts.append(ExpertA_v2(splited_dim,splited_dim))
        self.experts.append(ExpertB_v2(splited_dim,splited_dim))
        self.experts.append(ExpertC_v2(splited_dim,splited_dim))
        self.experts.append(ExpertD_v2(splited_dim,splited_dim))

        self.norm0 = GRN_3D(splited_dim,mode="Second")
        self.norm1 = GRN_3D(dim)

        
        self.split_num = split_num
    
    def forward(self,x):

        B,D,H,W,C = x.size()

        x_values = self.gate(x)
        x_values = self.soft(x_values)
        # B 768 8 8 8
        x = x.permute(0,4,1,2,3).contiguous()

        x = x.view(B,C//self.split_num,self.cubedim,self.cubedim,self.cubedim,D,H,W)
        x = x.permute(0,1,2,5,3,6,4,7).contiguous()
        x = x.view(B,C//self.split_num,self.cubedim*D,self.cubedim*H,self.cubedim*W)
        x = self.norm0(x)
        # 变成 B 12 32 32 32
        # 通道减八倍 特征图大小翻倍
        # B 8 8 8 4 

        # # B 32 32 32 12×4
        conv_x_ls = [expert(x).permute(0,2,3,4,1).contiguous() for expert in self.experts]
        B_,D_,H_,W_,C_ = conv_x_ls[0].size()


        for i in range(self.num_experts):
            conv_x_ls[i] = conv_x_ls[i].view(B_,D_//self.cubedim,self.cubedim,H_//self.cubedim,self.cubedim,W_//self.cubedim,self.cubedim,C_)
            conv_x_ls[i] = conv_x_ls[i].permute(0,1,3,5,2,4,6,7).contiguous()
            conv_x_ls[i] = conv_x_ls[i].view(B,D,H,W,C)
            # conv_x_ls[i] = self.norm0(conv_x_ls[i])

        results = x_values[:,:,:,:,0].unsqueeze(-1) * conv_x_ls[0]
        for i in range(1,self.num_experts):
            results += x_values[:,:,:,:,i].unsqueeze(-1) * conv_x_ls[i]
        
        results = self.norm1(results)
        return results




# 单独的一个lora layer
class Lora(nn.Module):
    def __init__(self,dim=768,r=4) -> None:
        super().__init__()
        self.lora_layer = nn.Sequential(
            nn.Linear(dim,r,bias=False),
            nn.Linear(r,dim,bias=False),
        )

        nn.init.kaiming_uniform_(self.lora_layer[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_layer[1].weight)
    
    def forward(self,x):
        return self.lora_layer(x)
        

# 相当于同质集成
class LoraMoe(nn.Module):
    def __init__(self,num_experts=4,dim=768,r=4) -> None:
        super().__init__()

        self.num_experts= num_experts

        self.lora_layers_q = nn.ModuleList([Lora(dim,r) for _ in range(num_experts)])
        self.lora_layers_v = nn.ModuleList([Lora(dim,r) for _ in range(num_experts)])

        self.gate_q = nn.Linear(dim,num_experts,bias=False)
        self.gate_v = nn.Linear(dim,num_experts,bias=False)
        self.soft = nn.Softmax(dim=-1)

        # 参数初始化
        nn.init.kaiming_uniform_(self.gate_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.gate_v.weight, a=math.sqrt(5))
    
    def forward(self,x):
        
        # B 8 8 8 num_experts
        gate_value_q = self.soft(self.gate_q(x))
        gate_value_v = self.soft(self.gate_v(x)) 

        # B L dim num_experts
        q_ls = [lora_layer(x) for lora_layer in self.lora_layers_q]
        v_ls = [lora_layer(x) for lora_layer in self.lora_layers_v]

        # Lora experts output   B L dim num_experts
        q_experts = torch.stack(q_ls,dim=-1)
        v_experts = torch.stack(v_ls,dim=-1)

        # B num_experts
        q_divisity = self.cal_divisity_v2(q_experts)
        v_divisity = self.cal_divisity_v2(v_experts)

        # B 1 1 1 num_experts
        q_divisity = q_divisity.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        v_divisity = v_divisity.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # B 8 8 8 num_experts
        q_value = gate_value_q + q_divisity
        v_value = gate_value_v + v_divisity

        new_q = torch.zeros_like(x)
        new_v = torch.zeros_like(x)

        for i in range(self.num_experts):
            new_q += q_value[:,:,:,:,i].unsqueeze(-1) * q_experts[:,:,:,:,:,i]
            new_v += v_value[:,:,:,:,i].unsqueeze(-1) * v_experts[:,:,:,:,:,i]

        return new_q,new_v
        

    def cal_divisity_v2(self,x):
        B,D,H,W,C,num_experts = x.size()

        # B L C num_experts
        x = x.view(B,D*H*W,C,num_experts).contiguous()
        # B C L num_experts
        x = x.permute(0,2,1,3).contiguous()
        x_norm = torch.norm(x,dim=-2,p=2,keepdim=True)

        x = x/(x_norm+1e-6)

        x_smilar = torch.einsum("bijm,bkjm->bikm",x,x)

        return torch.sum((1-x_smilar)/2,dim=(1,2))/(C*C)




class SEAdapeter(nn.Module):
    
    def __init__(self, inplanes, planes, hidden_channels, stride=1,
                 downsample=None, reduction=16):
        
        super(SEAdapeter, self).__init__()
        # 768
        self.ly_norm = nn.LayerNorm(inplanes)
        self.conv1 = nn.Conv3d(inplanes,hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.conv2 = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_channels,
            bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_channels)
        self.conv3 = nn.Conv3d(
            hidden_channels, planes , kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes )
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.se = SELayer(planes , reduction)
        self.stride = stride

        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        )
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.ly_norm(x)

        x = x.permute(0,4,1,2,3)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        out = out.permute(0,2,3,4,1).contiguous()

        return out

class SEAdapeterV2(nn.Module):
    def __init__(self,dim=768,norm_layer=nn.LayerNorm,hidden_channels=384,act_layer=nn.GELU) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim)

        self.adpter_channels = hidden_channels

        self.linear_down = nn.Linear(dim,hidden_channels,bias=False)

        self.linear_up = nn.Linear(hidden_channels,dim,bias=False)
        self.adapter_conv = nn.Conv3d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False)
        self.act = act_layer()
    
    def forward(self,x):
        
        shortcut = x 
        x = self.norm1(x)
        x = self.linear_down(x)
        x = x.permute(0,4,1,2,3)
        x = self.adapter_conv(x)
        x = x.permute(0,2,3,4,1).contiguous()
        x = self.act(x)
        x = self.linear_up(x)
        x = x + shortcut
        return x



class Adapter_former(nn.Module):
    def __init__(self,learnable_scale=True,dim=768,hidden_dim=4,norm_layer=nn.LayerNorm,
                 act_fn = nn.ReLU(inplace=True),before_norm=True,scale=1.0,drop=0.1) -> None:
        super().__init__()

        self.down = nn.Linear(dim,hidden_dim)
        self.up = nn.Linear(hidden_dim,dim)
        self.norm = norm_layer(dim)
        self.act_fn = act_fn
        self.before_norm = before_norm
        self.drop = nn.Dropout(drop)
        
        if learnable_scale:
            self.scale = torch.nn.Parameter(torch.ones(1))
        else:  
            self.scale = scale
        
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)
    
    def forward(self,x,residual=False):

        if residual:
            shortcut = x 

        if self.before_norm:
            x = self.norm(x)

        x = self.down(x)
        x = self.act_fn(x)
        x = self.drop(x)
        up = self.up(x)
        up = up * self.scale

        if not self.before_norm:
            up = self.norm(up)

        if residual:
            up = up + shortcut
        
        return up

if __name__ == "__main__":
  
    # model = getattr(nn,"LayerNorm")(768)
    # print(model)
    # model = LoraMoe()
    model = GITCrossAdapter(in_channels=768)
    # model = Conv_adapter_moe_v2(split_num=8)
    params = sum([p.numel() for p in model.parameters()])
    print(params/1e6)
    # t = torch.randn(2,8,8,8,768)
    # txt = torch.randn(2,1024)
    # y = model(t,txt)
    # print(y.shape)
    # new_q,new_v = model(t)
    # print(new_q.shape,new_v.shape)