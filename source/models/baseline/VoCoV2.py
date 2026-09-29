# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
import argparse
import torch.nn.functional as F
import math
from math import ceil
from typing import Optional
from models.baseline.ConvMoeV2 import MoEConv




class projection_head(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input):
        if torch.is_tensor(input):
            x = input
        else:
            x = input[-1]
            b = x.size()[0]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

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

class Swin(nn.Module):
    def __init__(self):
        super(Swin, self).__init__()
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        self.swinViT = SwinViT(
            in_chans=1,
            embed_dim=96,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
            use_v2=True,
        )
        norm_name = 'instance'
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=96,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=2 * 96,
            out_channels=2 * 96,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=4 * 96,
            out_channels=4 * 96,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=16 * 96,
            out_channels=16 * 96,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.trans = nn.Sequential(
            nn.Conv3d(in_channels=4,out_channels=8,kernel_size=7,padding=3),
            nn.BatchNorm3d(8),
            nn.Conv3d(in_channels=8,out_channels=1,kernel_size=3,padding=1)
        )

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=3,
        #     in_channels=16 * 96,
        #     out_channels=8 * 96,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=3,
        #     in_channels=96 * 8,
        #     out_channels=96 * 4,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder3 = UnetrUpBlock(
        #     spatial_dims=3,
        #     in_channels=96 * 4,
        #     out_channels=96 * 2,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        # self.decoder2 = UnetrUpBlock(
        #     spatial_dims=3,
        #     in_channels=96 * 2,
        #     out_channels=96,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder1 = UnetrUpBlock(
        #     spatial_dims=3,
        #     in_channels=96,
        #     out_channels=96,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )

    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x_in):
        b = x_in.size()[0]
        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(self.trans(x_in))
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        encs = [enc0, enc1, enc2, enc3, dec4]

        # for enc in encs:
        #     print(enc.shape)

        out = self.forward_encs(encs)
        return out.view(b, -1)
    
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

class Cls_net(nn.Module):
    def __init__(self,in_dim=2304,hidden_dim=512,classses=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_dim,out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,out_features=hidden_dim//2),
            nn.Linear(in_features=hidden_dim//2,out_features=classses)
        )
    def forward(self,x):
        return self.fc(x) 

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

class VoCoV2(nn.Module):
    def __init__(self, spatial_dims=3, num_classes=2, r=8, ConvMoe=False):
        super().__init__()

        model = Swin()

        pretrained_path = "/home/cyx/Datasets/VoCo_L_SSL_head.pt"
        model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict,strict=False)

        for p in model.parameters():
            p.requires_grad = False
        image_encoder = model.swinViT

        # for p in image_encoder.parameters():
        #     p.requires_grad = False


        image_encoder.patch_embed = MultiModalPatchEmbed(image_encoder.patch_embed, After_fusion())


        if model.trans is not None:
            for k in model.trans.parameters():
                k.requires_grad = True

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

        model.swinViT = image_encoder

        self.image_encoder = model
        self.cls = Cls_net()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))

        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x):
        x = self.image_encoder(x)  # B D
        return x 
        y = self.cls(x)
        return y




if __name__ == '__main__':
    model = VoCoV2()
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
    # x = torch.randn(4,1,128,128,128)
    # print(model(x).shape)


