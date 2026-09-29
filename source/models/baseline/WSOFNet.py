import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Tuple
import torch.nn.functional as F
from visualizer import get_local

def sub_opm(f_f,f1):
    f_m_p = ((f_f*f1)/sum(f_f*f_f))*f_f
    return f1 - f_m_p

def OPM(f_f,f1,f2,f3,f4):
    B,C,D,H,W = f_f.shape

    f_f = f_f.view(B,C,-1).contiguous()
    f1 = f1.view(B,C,-1).contiguous()
    f2 = f2.view(B,C,-1).contiguous()
    f3 = f3.view(B,C,-1).contiguous()
    f4 = f4.view(B,C,-1).contiguous()

    f1 = sub_opm(f_f,f1)
    f2 = sub_opm(f_f,f2)
    f3 = sub_opm(f_f,f3)
    f4 = sub_opm(f_f,f4)

    f1 = f1.view(B,C,D,H,W).contiguous()
    f2 = f2.view(B,C,D,H,W).contiguous()
    f3 = f3.view(B,C,D,H,W).contiguous()
    f4 = f4.view(B,C,D,H,W).contiguous()

    return f1,f2,f3,f4

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1).contiguous().float(),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_d: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int, int],
        k_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_d, q_h, q_w = q_size
    k_d, k_h, k_w = k_size

    Rd = get_rel_pos(q_d, k_d, rel_pos_d)
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, h, _,dim = q.shape
    r_q = q.reshape(B*h, q_d, q_h, q_w, dim)

    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)

    attn = (
            attn.view(B*h, q_d, q_h, q_w, k_d, k_h, k_w) + rel_d[:, :, :, :, None, None] + rel_h[:, :, :, None, :,
                                                                                         None] + rel_w[:, :, :, None,
                                                                                                 None, :]
    ).view(B, h, q_d * q_h * q_w, k_d * k_h * k_w)

    return attn


class CNN_Block(nn.Module):
    def __init__(self,in_chans,out_chans,kernel_size=3,is_act=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_chans,out_chans,kernel_size=kernel_size,padding=kernel_size//2),
            nn.BatchNorm3d(out_chans),
            nn.GELU() if is_act else nn.Identity()
        )
    def forward(self,x):
        return self.layer(x)


class PEM(nn.Module):
    def __init__(self,in_chans,out_chans):
        super().__init__()
        self.layer = nn.Sequential(
            CNN_Block(in_chans,out_chans,kernel_size=1,is_act=True),
            CNN_Block(out_chans,out_chans),
            CNN_Block(out_chans,out_chans,kernel_size=1)
        )
    def forward(self,x):
        return self.layer(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class MBConv(nn.Module):
    def __init__(self, inp, oup, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool3d(3, 2, 1)
            self.proj = nn.Conv3d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv3d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm3d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw, self.id = image_size

        self.heads = heads


        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.rel_pos_d = nn.Parameter(torch.zeros(2 * self.ih - 1, dim_head))
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * self.iw - 1, dim_head))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * self.id - 1, dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = add_decomposed_rel_pos(attn, q,self.rel_pos_d, self.rel_pos_h, self.rel_pos_w, (self.ih, self.iw, self.id),(self.ih, self.iw, self.id))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size=(8,8,8), heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw, self.id = image_size
        # self.downsample = downsample
        #
        # if self.downsample:
        #     self.pool1 = nn.MaxPool2d(3, 2, 1)
        #     self.pool2 = nn.MaxPool2d(3, 2, 1)
        #     self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.res = nn.Conv3d(inp,oup,kernel_size=1)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw id -> b (ih iw id) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw id) c -> b c ih iw id', ih=self.ih, iw=self.iw, id=self.id)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw id-> b (ih iw id) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw id) c -> b c ih iw id', ih=self.ih, iw=self.iw, id=self.id)
        )

    def forward(self, x):
        # if self.downsample:
        #     x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        # else:
        x = self.res(x) + self.attn(x)
        x = x + self.ff(x)
        return x

class FEM(nn.Module):
    def __init__(self,in_chans,ch):
        super().__init__()
        self.conv_layer0 = nn.Sequential(
            CNN_Block(in_chans,ch,is_act=True),
            CNN_Block(ch,ch,is_act=True)
        )

        self.res_layer1 = nn.Conv3d(ch, ch*2, kernel_size=1)
        self.conv_layer1 = nn.Sequential(
            MBConv(ch, ch),
            MBConv(ch, ch*2)
        )

        self.res_layer2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3,stride=4,padding=1),
            nn.Conv3d(ch*2, ch*4,kernel_size=1)
        )
        self.conv_layer2 = nn.Sequential(
            MBConv(ch*2, ch*2, downsample=False),
            MBConv(ch*2, ch * 2, downsample=True),
            MBConv(ch*2, ch * 4, downsample=True)
        )

        input_chans = ch*4
        self.trans = nn.ModuleList()
        self.trans.append(Transformer(input_chans,input_chans))

        self.trans.extend(
            [
                Transformer(inp=input_chans+2*i*ch, oup=input_chans+2*(i+1)*ch)
                for i in range(6)
            ]
        )

    def forward(self,x):
        x = self.conv_layer0(x)
        x = self.res_layer1(x) + self.conv_layer1(x)
        x = self.res_layer2(x) + self.conv_layer2(x)

        for trans in self.trans:
            x = trans(x)

        return x

class AMF2M(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Linear(dim, dim)
        self.fc_pr = nn.Linear(dim, dim*4)

        self.conv = nn.Conv3d(2,1,kernel_size=5,padding=2)
        self.conv_pr = nn.Conv3d(1,4,kernel_size=3,padding=1)


    def forward(self, f1, f2, f3, f4):
        B,C,H,W,D = f1.shape
        f_cat = torch.cat([f1,f2,f3,f4],dim=1)
        f_sum = f1 + f2 + f3 + f4

        fc_avg = self.avg_pool(f_sum).view(f_sum.size(0),-1).contiguous()
        fc_max = self.max_pool(f_sum).view(f_sum.size(0),-1).contiguous()
        f_ch = torch.sigmoid(self.fc(fc_max)+self.fc(fc_avg))

        avg_out = torch.mean(f_sum,keepdim=True,dim=1)
        max_out,_ = torch.max(f_sum,keepdim=True,dim=1)
        f_sp = torch.cat([avg_out,max_out],dim=1)
        f_sp = torch.sigmoid(self.conv(f_sp))

        f_sp = self.conv_pr(f_sp)
        f_sp = f_sp.permute(0,2,3,4,1).contiguous()
        f_sp = f_sp.view(f_sp.size(0),-1,4)

        # f_sp = f_sp.view(f_sp.size(0), -1).contiguous()

        f_ch = self.fc_pr(f_ch).view(f_ch.size(0),-1,4)
        # f_sp = self.fc_pr2(f_sp).view(f_sp.size(0),-1,4)

        f_fusion = torch.einsum("b h d, b w d -> b h w",f_ch,f_sp)
        f_fusion = torch.sigmoid(f_fusion)

        f_fusion = torch.softmax(f_fusion,dim=-1).view(B,-1,H,W,D)
        f_fusion = f_fusion.repeat(1, 4, 1, 1, 1)

        f_hybrid = f_cat * f_fusion
        f_hybrid = f_hybrid.view(B,-1,4,D,H,W)
        f_hybrid = torch.sum(f_hybrid, dim=2)

        return f_hybrid

class wsofNet(nn.Module):
    def __init__(self, in_chans,dim=72, hidden_dim=128, num_classes=2, ch1=16, ch2=32):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_chans,dim//2,kernel_size=3,stride=2,padding=1),
            torch.nn.GELU(),
            nn.Conv3d(dim//2,dim,kernel_size=3,stride=2,padding=1)
        )
        # Because the net was revised to 3D version, to reduce GPU memory spending we utilize a patch embedd module like other SOTA methods(RepVit)
        self.pem_layers = nn.ModuleList(
            [
                PEM(dim//4,out_chans=2*hidden_dim)
                for i in range(4)
            ]
        )

        self.pem_layers2 = nn.ModuleList(
            [
                PEM(2*hidden_dim, 2*hidden_dim)
                for i in range(4)
            ]
        )

        self.amf2m = nn.ModuleList(
            [
                AMF2M(dim=2*hidden_dim),
                AMF2M(dim=2*hidden_dim)
            ]
        )

        self.fem_layers = nn.ModuleList(
            [
                FEM(in_chans=2*hidden_dim, ch=ch1),
                FEM(in_chans=2*hidden_dim, ch=ch2)
            ]
        )

        self.classifier = nn.Linear(16*(ch1+ch2),num_classes)
        self.pool = nn.AdaptiveAvgPool3d(1)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.001)
            elif isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out")

    # @get_local("temp")
    def forward(self,x):
        x = self.patch_embed(x)
        f1,f2,f3,f4 = torch.chunk(x,chunks=4,dim=1)


        ls = [f1,f2,f3,f4]

        for i in range(4):
            ls[i] = self.pem_layers[i](ls[i])

        f1_f = self.amf2m[0](ls[0],ls[1],ls[2],ls[3])

        ls[0],ls[1],ls[2],ls[3] = OPM(f1_f,ls[0],ls[1],ls[2],ls[3])

        for i in range(4):
            ls[i] = self.pem_layers2[i](ls[i])
        f2_f = self.amf2m[1](ls[0],ls[1],ls[2],ls[3])
        f1_f = self.fem_layers[0](f1_f)
        f2_f = self.fem_layers[1](f2_f)
        f_final = torch.cat([f1_f,f2_f],dim=1)
        f_final = self.pool(f_final).view(f_final.size(0),-1)
        # temp = f_final
        y = self.classifier(f_final)
        return y

if __name__ == "__main__":
    t = torch.randn(4,4,128,128,128)
    model = wsofNet(in_chans=4,dim=96*4)
    print(model(t).shape)
    # 29.27M
    params = sum([param.nelement() for param in model.parameters()])/1e6
    print(params)