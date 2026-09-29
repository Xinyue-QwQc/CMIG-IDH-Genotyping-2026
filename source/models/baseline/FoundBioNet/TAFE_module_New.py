# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from  math  import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
import math
from models.baseline.ConvMoeV2 import MoEConv
# from .ConvMoe import MoEConv
from typing import Optional
from visualizer import get_local


class After_fusion(nn.Module):
    def __init__(self,in_chans=48*4, hidden_chans=256, out_chans=48, activation=nn.LeakyReLU(inplace=True),norm_layer=nn.BatchNorm3d):
        super().__init__()

        self.down = nn.Conv3d(in_chans,hidden_chans,kernel_size=1)
        self.norm0 = norm_layer(hidden_chans)

        self.up = nn.Conv3d(hidden_chans, out_chans,kernel_size=1)
        self.norm1 = norm_layer(out_chans)

        self.act_fn = activation

        # 初始化参数的方式

    def forward(self,x):

        x = self.down(x)
        x = self.norm0(x)
        x = self.act_fn(x)

        x = self.up(x)
        x = self.norm1(x)
        x = self.act_fn(x)

        return x

def process_window(x,window_num):
    # window_d = ceil(math.pow(window_num,1/3))
    BW,W_num,dim = x.shape

    x = x.view(BW,7,7,7,dim)
    # x = x.permute(0,4,1,2,3).contiguous()
    # x = x.view(BW//window_num,window_d*7,window_d*7,window_d*7,dim)
    return x 

def reverse_window(x,window_num):
    # window_d = ceil(math.pow(window_num,1/3))
    B,D,H,W,C = x.shape
    # x = x.view(B,D//window_d,window_d,H//window_d,window_d,W//window_d,window_d,C)
    # x = x.permute(0,1,3,5,2,4,6,7).contiguous()
    x = x.view(B,-1,C)
    return x 



class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_q: nn.Module,
            linear_b_v: nn.Module,
            linear_a_k: Optional[nn.Module] = None,
            linear_b_k: Optional[nn.Module] = None,
            r=4,
            conv_lora=False,
            window_num = 100,
            raw_shape = 10
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        if linear_a_k is not None:
            self.linear_a_k = linear_a_k
            self.linear_b_k = linear_b_k
        else:
            self.linear_a_k = None
        self.dim = qkv.in_features
        self.conv_lora = conv_lora
        self.window_num = window_num
        self.raw_shape = raw_shape 

        if self.conv_lora:
            self.conv_q = MoEConv(d=r, scales=[1, 2, 4, 8], M=4, K=2)
            self.conv_v = MoEConv(d=r, scales=[1, 2, 4, 8], M=4, K=2)
            if linear_a_k is not None:
                self.conv_k = MoEConv(d=r, scales=[1, 2, 4, 8], M=4, K=2)

    def forward(self, x):
        qkv = self.qkv(x)  # B*window_num window_size dim  x 
        
        if self.conv_lora:
            x = process_window(x,self.window_num)

            new_q = self.linear_b_q(self.conv_q(self.linear_a_q(x)))
            new_v = self.linear_b_v(self.conv_v(self.linear_a_v(x)))

            new_q = reverse_window(new_q,self.window_num)
            new_v = reverse_window(new_v,self.window_num)
        else:
            new_q = self.linear_b_q(self.linear_a_q(x))
            new_v = self.linear_b_v(self.linear_a_v(x))

        if self.linear_a_k is not None:
            if self.conv_lora:
                new_k = self.linear_b_k(self.conv_k(self.linear_a_k(x)))
                new_k = reverse_window(new_k,self.window_num)
            else:
                new_k = self.linear_b_k(self.linear_a_k(x))

            qkv[:, :, self.dim:-self.dim] += new_k


        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv

class MultiModalPatchEmbed(nn.Module):
    def __init__(self,old_embed,after_fusion):
        super().__init__()
        self.patch_embed = old_embed
        self.after_fusion = after_fusion

    def forward(self,x):
        C = x.shape[1]
        ls = []
        # B D H W C
        for i in range(C):
            ls.append(self.patch_embed(x[:,i,:,:,:].unsqueeze(1)))
        x_ = torch.cat(ls,dim=1)
        x = self.after_fusion(x_)
        return  x
    


class BrainSegFounder(nn.Module):
    def __init__(self, spatial_dims=3, num_classes=2, r=8, ConvMoe=False):
        super().__init__()

        patch_size = ensure_tuple_rep(2, spatial_dims)  # YY [2,2,2]
        window_size = ensure_tuple_rep(7, spatial_dims)  # YY [7,7,7]
        # dim = bottleneck_depth                                     # YY 768 or 1536
        image_encoder = SwinViT(
            in_chans=1,  # 2 for T1T2
            embed_dim=48,  # YY 48, 96
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],  # YY [2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],  # YY [3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
        )

        check = torch.load("/home/cyx/Datasets/64-gpu-model_bestValRMSE.pt")
        model_check = check["state_dict"]
        for k in list(model_check.keys()):
            if "swinViT" in k:
                model_check[k[len("module.swinViT."):]] = model_check[k]
            del model_check[k]

        msg = image_encoder.load_state_dict(model_check,strict=False)

        for p in image_encoder.parameters():
            p.requires_grad = False

        image_encoder.patch_embed = MultiModalPatchEmbed(image_encoder.patch_embed,After_fusion())

        #  存 lora的参数
        self.w_As = []
        self.w_Bs = []
        self.conv_As = []
        num_windows = [1000,125,27,8]
        raw_shape = [64,32,16,8]

        for j in range(4):
            layer = getattr(image_encoder,f"layers{j+1}")
            # 增加lora adapter
            for t_layer_i, blk in enumerate(layer[0].blocks):
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                w_a_linear_k = nn.Linear(self.dim, r, bias=False)
                w_b_linear_k = nn.Linear(r, self.dim, bias=False)

                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)

                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)

                blk.attn.qkv = _LoRA_qkv(w_qkv_linear,
                                         w_a_linear_q, w_a_linear_v,
                                         w_b_linear_q, w_b_linear_v,
                                         w_a_linear_k, w_b_linear_k,
                                         r=r, conv_lora=ConvMoe,window_num=num_windows[j])

        self.reset_parameters()

        self.image_encoder = image_encoder

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))

        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x):
        x_size = x.size(0)
        x = self.image_encoder(x)  # B D
        x = x[-1]
        x_pooled = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x_size, -1)
        return x_pooled



if __name__ == "__main__":
    model = BrainSegFounder()
    print(model)

    total = 0
    all_params = 0
    for name,values in model.named_parameters():
        if values.requires_grad:
            print(name)
            total += values.nelement()
        all_params += values.nelement()
    print(total/1e6)
    print(total*100/all_params)
    print("ALL",all_params/1e6)
    t = torch.randn(1,4,128,128,128)
    print(model(t).shape)