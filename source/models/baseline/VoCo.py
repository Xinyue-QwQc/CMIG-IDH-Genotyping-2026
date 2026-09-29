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
from math import ceil

import torch
import torch.nn as nn

# from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.networks.nets import SwinUNETR
from monai.utils import ensure_tuple_rep
import math
from models.baseline.ConvMoeV2 import MoEConv
# from .ConvMoe import MoEConv
from typing import Optional


# from visualizer import get_local


class After_fusion(nn.Module):
    def __init__(self, in_chans=96 * 4, hidden_chans=256, out_chans=96, activation=nn.LeakyReLU(inplace=True),
                 norm_layer=nn.BatchNorm3d):
        super().__init__()

        self.down = nn.Conv3d(in_chans, hidden_chans, kernel_size=1)
        self.norm0 = norm_layer(hidden_chans)

        self.up = nn.Conv3d(hidden_chans, out_chans, kernel_size=1)
        self.norm1 = norm_layer(out_chans)

        self.act_fn = activation

        # 初始化参数的方式

    def forward(self, x):
        x = self.down(x)
        x = self.norm0(x)
        x = self.act_fn(x)

        x = self.up(x)
        x = self.norm1(x)
        x = self.act_fn(x)

        return x


def process_window(x, window_num):
    #
    BW, W_num, dim = x.shape
    window_d = ceil(math.pow(W_num, 1 / 3))
    x = x.view(BW, window_d, window_d, window_d, dim)
    # x = x.permute(0,4,1,2,3).contiguous()
    # x = x.view(BW//window_num,window_d*7,window_d*7,window_d*7,dim)
    return x


def reverse_window(x, window_num):
    # window_d = ceil(math.pow(window_num,1/3))
    B, D, H, W, C = x.shape
    # x = x.view(B,D//window_d,window_d,H//window_d,window_d,W//window_d,window_d,C)
    # x = x.permute(0,1,3,5,2,4,6,7).contiguous()
    x = x.view(B, -1, C)
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
            window_num=100,
            raw_shape=10
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
            x = process_window(x, self.window_num)

            new_q = self.linear_b_q(self.conv_q(self.linear_a_q(x)))
            new_v = self.linear_b_v(self.conv_v(self.linear_a_v(x)))

            new_q = reverse_window(new_q, self.window_num)
            new_v = reverse_window(new_v, self.window_num)
        else:
            new_q = self.linear_b_q(self.linear_a_q(x))
            new_v = self.linear_b_v(self.linear_a_v(x))

        if self.linear_a_k is not None:
            if self.conv_lora:
                new_k = self.linear_b_k(self.conv_k(self.linear_a_k(x)))
                new_k = reverse_window(new_k, self.window_num)
            else:
                new_k = self.linear_b_k(self.linear_a_k(x))

            qkv[:, :, self.dim:-self.dim] += new_k

        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


class MultiModalPatchEmbed(nn.Module):
    def __init__(self, old_embed, after_fusion):
        super().__init__()
        self.patch_embed = old_embed
        self.after_fusion = after_fusion

    def forward(self, x):
        C = x.shape[1]
        ls = []
        # B D H W C
        for i in range(C):
            ls.append(self.patch_embed(x[:, i, :, :, :].unsqueeze(1)))
        x_ = torch.cat(ls, dim=1)
        x = self.after_fusion(x_)
        return x


class Cls_net(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm3d, pool_layer=nn.AdaptiveAvgPool3d, input_channels=1536,
                 num_classes=2) -> None:
        super().__init__()

        self.bn0 = norm_layer(input_channels)
        self.pool = pool_layer(1)
        self.ly0 = nn.Sequential(
            nn.Linear(in_features=input_channels, out_features=input_channels // 2),
            nn.Linear(in_features=input_channels // 2, out_features=num_classes)
        )

    # @get_local("temp")
    def forward(self, x):
        x = self.bn0(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # temp = x
        y = self.ly0(x)
        return y


def load(model, model_dict):
    # make sure you load our checkpoints
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    else:
        state_dict = model_dict
    current_model_dict = model.state_dict()
    for k in current_model_dict.keys():
        if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
            print(k)
    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else
        current_model_dict[k]
        for k in current_model_dict.keys()}
    model.load_state_dict(new_state_dict, strict=True)
    return model


class VoCo(nn.Module):
    def __init__(self, spatial_dims=3, num_classes=2, r=4, ConvMoe=True):
        super().__init__()

        model = SwinUNETR(img_size=(96, 96, 96),
                          in_channels=1,
                          out_channels=21,
                          feature_size=96,
                          use_v2=True)

        pretrained_path = "/home/cyx/Datasets/VoComni_L.pt"
        model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        model = load(model, model_dict)
        image_encoder = model.swinViT

        for p in image_encoder.parameters():
            p.requires_grad = False

        image_encoder.patch_embed = MultiModalPatchEmbed(image_encoder.patch_embed, After_fusion())

        #  存 lora的参数
        self.w_As = []
        self.w_Bs = []
        self.conv_As = []
        num_windows = [1000, 125, 27, 8]
        raw_shape = [64, 32, 16, 8]

        for j in range(4):
            layer = getattr(image_encoder, f"layers{j + 1}")
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
                                         r=r, conv_lora=ConvMoe, window_num=num_windows[j])

        self.reset_parameters()

        self.image_encoder = image_encoder
        self.cls = Cls_net()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))

        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x):
        x = self.image_encoder(x)  # B D
        x = x[-1]
        y = self.cls(x)
        return y


if __name__ == "__main__":
    model = VoCo()
    print(model)

    t = torch.randn(1,4,96,96,96)

    total = 0
    all_params = 0
    for name, values in model.named_parameters():
        if values.requires_grad:
            print(name)
            total += values.nelement()
        all_params += values.nelement()
    print(total / 1e6)
    print(total * 100 / all_params)
    print("ALL", all_params / 1e6)
    # t = torch.randn(1, 4, 128, 128, 128)
    print(model(t).shape)