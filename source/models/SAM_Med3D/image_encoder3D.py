# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Type,Union
from .model_util import _Fact_tt, _Fact_tk,After_confusion,Project_layer
from .PEFT_module import SEAdapeter,SEAdapeterV2,Adapter_former,LoraMoe,Conv_adapter_moe,Conv_adapter_moe_v2
from .mismatch_branch import  Project_kv,Gate,Mismatch_extractor,Mismatch_extractorV2
from .Lora_ssm import  Mamba_adapter
# import monai.networks.nets as nets

# nets.swin_unetr
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    
class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class Mis_sign_getter(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample_stride=1,num_heads=12,
                 kv_bias=True, use_rel_pos=True, hidden_dim: Optional[int] = None,
                 input_size: Optional[Tuple[int, int]] = None, is_up=False):
        super().__init__()
        self.project_kv = Project_kv(in_channels, out_channels, down_sample_stride, kv_bias,hidden_dim,up_sample=is_up)
        # self.mis_scale = nn.Parameter(torch.tensor(1.0))
        if hidden_dim is None:
            self.proj = nn.Linear(out_channels,out_channels)
        else:
            self.proj = Project_layer(out_channels,hidden_channels=hidden_dim,out_channels=out_channels,bias=False)


        # self.gate_mis = Gate(out_channels)

        head_dim = out_channels // num_heads

        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        self.use_rel_pos = use_rel_pos

        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x, sam_q):
        B, _, _, _, _ = x.shape

        kv = self.project_kv(x)

        k, v = kv.reshape(2, B * self.num_heads, 512, -1).unbind(0)

        attn = (sam_q * self.scale) @ k.transpose(-2,-1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, sam_q, self.rel_pos_d, self.rel_pos_h, self.rel_pos_w, (8,8,8), (8,8,8))

        attn = attn.softmax(dim=-1)
        mis_x = (attn @ v).view(B, self.num_heads, 8, 8, 8,-1).permute(0, 2, 3, 4, 1, 5).reshape(B, 8, 8, 8, -1)

        mis_x = self.proj(mis_x)

        # gate_mis = self.gate_mis(mis_x)  * gate_mis * self.mis_scale

        return mis_x



# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT3D(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        after_fusion_channels = 128,
        conv_adapter = False,
        adapt_former = False,
        parallel = True,
        Mis_branch = False
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        #---------------------------------------<New to SAM Med 3D>--------------------------------------------------
        self.global_attn_indexes = global_attn_indexes
        self.mis_branch = Mis_branch

        if self.mis_branch:
            # self.mis_extractor = Mismatch_extractor()
            self.mis_extractor = Mismatch_extractorV2()

        # ---------------------------------------<New to SAM Med 3D>--------------------------------------------------



        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None

        self.after_confusion = After_confusion(hidden_channels=after_fusion_channels)
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size),
                conv_adapter = conv_adapter,
                adapter_former = adapt_former,
                parallel = parallel,
                mis_qkv = True if i in global_attn_indexes and Mis_branch else False,
                depth_index = i
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            # nn.LayerNorm(out_chans),
            LayerNorm3d(out_chans),
            nn.Conv3d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
            # nn.LayerNorm(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input_size = [1,1,256,256,256]    input size B 1 128 128 128
        # import IPython; IPython.embed()
        # ls = []

        # ---------------------------------------<New to SAM Med 3D>--------------------------------------------------
        if self.mis_branch:
            #  分别是T2 Flair模态数据
            T2_Flair_signs = self.mis_extractor(x[:,-2,...].unsqueeze(1),x[:,-1,...].unsqueeze(1))

        C = x.shape[1]
        ls = []
        # B D H W C
        for i in range(C):
            ls.append(self.patch_embed(x[:,i,:,:,:].unsqueeze(1)))
        x_ = torch.cat(ls,dim=-1)
        x = self.after_confusion(x_)

        # ---------------------------------------<New to SAM Med 3D>--------------------------------------------------

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for i, blk in enumerate(self.blocks):
            if i in self.global_attn_indexes and self.mis_branch:
                mis_index = (i-2)//3
                x = blk(x,T2_Flair_signs[mis_index])
            else:
                x = blk(x,None)
            # ls.append(x)
        # x = [1,16,16,16,768]  x' = B 8 8 8 768
        x = self.neck(x.permute(0, 4, 1, 2, 3).contiguous())

        # output_size = [1,256,16,16,16]    B 256 8 8 8 
        # return x,ls
        return x 


class Block3D(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int, int]] = None,
        conv_adapter = True,
        adapter_former = False,
        parallel = True,
        mis_qkv = False,
        depth_index = 0,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.depth_index = depth_index

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
            mis_qkv=mis_qkv,
            depth_index = depth_index
        )

        self.conv_adapter = conv_adapter
        self.is_adapter_former = adapter_former
        self.parallel = parallel
        self.depth_index = depth_index

        if conv_adapter and self.depth_index > 5:
            # self.conv_adapter0 = SEAdapeter(dim,dim,hidden_channels=128)
            # self.conv_adapter = SEAdapeter(dim,dim,hidden_channels=128)
            self.conv_adapter = Mamba_adapter(in_channels=dim,hidden_channels=128)
            # self.conv_adapter = Conv_adapter_moe(dim=dim,hidden_dim=64)
            # self.conv_adapter = SEAdapeterV2(dim=dim)
            # self.conv_adapter1 = SEAdapeterV2(dim=dim)
        if adapter_former:
            self.adapter_former = Adapter_former(dim=dim,hidden_dim=64)
        
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor, mis_feature:Optional[Union[torch.Tensor, None]]) -> torch.Tensor:

        if self.conv_adapter and self.depth_index>5:
            x = self.conv_adapter(x)

        
        shortcut = x
        # lora conv adapter v2
        # if self.conv_adapter:
        #     x = self.conv_adapter0(x)
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            D, H, W = x.shape[1], x.shape[2], x.shape[3]
            x, pad_dhw = window_partition3D(x, self.window_size)

        x = self.attn(x,mis_feature)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition3D(x, self.window_size, pad_dhw, (D, H, W))

        x = shortcut + x

        # 此时是sequential
        # if self.conv_adapter:
        #     if self.parallel:
        #         adapt_x = self.conv_adapter(x)
        #         mlp_x = self.mlp(self.norm2(x))
        #         x = x + mlp_x + adapt_x
        #     else:
        #         x = self.conv_adapter(x)
        #         x = x + self.mlp(self.norm2(x))
        # else:
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int, int]] = None,
        mis_qkv = False,
        depth_index = 0
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()

        # ---------------------------------------<New to SAM Med 3D>--------------------------------------------------
        # channels = [128, 256, 512, 512]
        # down_sample_strides = [4, 2, 1, 1]

        channels = [40, 80, 160, 320]
        down_sample_strides = [4, 2, 1, 2]
        # channels.reverse()
        # down_sample_strides.reverse()

        self.mis_qkv = mis_qkv
        if mis_qkv:
            mis_cfg_index = (depth_index -2) // 3
            self.mis_attention = Mis_sign_getter(channels[mis_cfg_index],dim,
                                 down_sample_stride = down_sample_strides[mis_cfg_index],
                                 input_size = input_size,
                                 num_heads=num_heads,
                                 use_rel_pos=use_rel_pos,
                                 hidden_dim=128,
                                 is_up=True if mis_cfg_index == 3 else False
                                 )
            self.proj_mismatch = Project_layer(in_channels=dim*2,hidden_channels=128,out_channels=dim,
                                               act_fn="gelu",norm_layer=nn.LayerNorm)
        # ---------------------------------------<New to SAM Med 3D>--------------------------------------------------

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x: torch.Tensor,
                mis_feature:Optional[Union[torch.Tensor, None]]) -> torch.Tensor:
        
        B, D, H, W, _ = x.shape
        # B D H W 3*dim
        # qkv with shape (3, B, nHead, H * W, C)

        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(0)

        # ---------------------------------------<New to SAM Med 3D>--------------------------------------------------
        if self.mis_qkv:
            mis_sign = self.mis_attention(mis_feature, q)
        # ---------------------------------------<New to SAM Med 3D>--------------------------------------------------

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_d, self.rel_pos_h, self.rel_pos_w, (D, H, W), (D, H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, D, H, W, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, -1)
        x = self.proj(x)

        if self.mis_qkv:
            final_signs = torch.cat([x,mis_sign],dim=-1)
            final_signs = self.proj_mismatch(final_signs)
            return final_signs

        return x


def window_partition3D(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, D, H, W, C = x.shape

    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    x = x.view(B, Dp // window_size, window_size, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Dp, Hp, Wp)


def window_unpartition3D(
    windows: torch.Tensor, window_size: int, pad_dhw: Tuple[int, int, int], dhw: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    B = windows.shape[0] // (Dp * Hp * Wp // window_size // window_size // window_size)
    x = windows.view(B, Dp // window_size, Hp // window_size, Wp // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp,  -1)

    if Hp > H or Wp > W or Dp > D:
        x = x[:, :D, :H, :W, :].contiguous()
    return x


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
    
    B, _, dim = q.shape
    r_q = q.reshape(B, q_d, q_h, q_w, dim)

    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)
    

    
    attn = (
        attn.view(B, q_d, q_h, q_w, k_d, k_h, k_w) + rel_d[:, :, :, :, None, None] + rel_h[:, :, :, None, :, None] + rel_w[:, :, :,None,None, :]
    ).view(B, q_d * q_h * q_w, k_d * k_h * k_w)

    return attn


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16, 16),
        stride: Tuple[int, int] = (16, 16, 16),
        padding: Tuple[int, int] = (0, 0, 0),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x

if __name__ == "__main__":
    image_encoder=ImageEncoderViT3D(
            depth=12,
            embed_dim=768,
            img_size=128,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2,5,8,11],
            window_size=14,
            out_chans=384)

