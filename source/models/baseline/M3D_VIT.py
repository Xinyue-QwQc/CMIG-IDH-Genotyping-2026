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

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
import math
from models.baseline.ConvMoe import MoEConv
# from .ConvMoe import MoEConv
from typing import Optional


class After_fusion(nn.Module):
    def __init__(self,in_chans=768*4, hidden_chans=128, out_chans=768, activation=nn.LeakyReLU(inplace=True),norm_layer=nn.LayerNorm):
        super().__init__()

        self.down = nn.Linear(in_chans,hidden_chans)
        self.norm0 = norm_layer(hidden_chans)

        self.up = nn.Linear(hidden_chans, out_chans)
        self.norm1 = norm_layer(out_chans)

        self.act_fn = activation

        # 初始化参数的方式
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # elif isinstance(m,nn.LayerNorm):


    def forward(self,x):

        x = self.down(x)
        x = self.norm0(x)
        x = self.act_fn(x)

        x = self.up(x)
        x = self.norm1(x)
        x = self.act_fn(x)

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
            conv_lora=False
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

        if self.conv_lora:
            self.conv_q = MoEConv(d=r, scales=[1, 2, 4, 8], M=4, K=2)
            self.conv_v = MoEConv(d=r, scales=[1, 2, 4, 8], M=4, K=2)
            if linear_a_k is not None:
                self.conv_k = MoEConv(d=r, scales=[1, 2, 4, 8], M=4, K=2)

    def forward(self, x):
        qkv = self.qkv(x)  # B D H W 3*dim  qkv

        if self.conv_lora:
            new_q = self.linear_b_q(self.conv_q(self.linear_a_q(x)))
            new_v = self.linear_b_v(self.conv_v(self.linear_a_v(x)))
        else:
            new_q = self.linear_b_q(self.linear_a_q(x))
            new_v = self.linear_b_v(self.linear_a_v(x))

        if self.linear_a_k is not None:
            if self.conv_lora:
                new_k = self.linear_b_k(self.conv_k(self.linear_a_k(x)))
            else:
                new_k = self.linear_b_k(self.linear_a_k(x))

            qkv[:, :, self.dim:-self.dim] += new_k

        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            # >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            # >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            # >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.after_confusion = After_fusion(hidden_chans=128)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x):
        ls = []

        for i in range(4):
            ls.append(self.patch_embedding(x[:,i,:,:,:].unsqueeze(1)))

        x_ = torch.cat(ls,dim=-1)
        x = self.after_confusion(x_)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])
        return x, hidden_states_out





class ViT3DTower(nn.Module):
    def __init__(self, in_chans, img_size, patch_size, select_features = "patch", select_layer=-1,  spatial_dims=3):
        super().__init__()
        # self.config = config
        self.select_layer = select_layer
        self.select_feature = select_features

        self.vision_tower = ViT(
            in_channels=in_chans,
            img_size=img_size,
            patch_size=patch_size,
            pos_embed="perceptron",
            spatial_dims=spatial_dims,
            classification=True,
        )

    def forward(self, images):
        last_feature, hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer == 1:
            image_features = torch.cat([hidden_states[2],last_feature],dim=-1)
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            # 对结果取mean
            image_features = torch.mean(image_features[:, 1:],dim=1,keepdim=False)
        elif self.select_feature == 'cls_patch':
            image_features = image_features[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

class M3D_VIT_CLS(nn.Module):
    def __init__(self, in_chans, img_size, patch_size, select_features = "cls_patch", select_layer=-1,
                 spatial_dims=3, num_classes=2, r=4, ConvMoe=True, text_prompt=False):
        super().__init__()

        image_encoder = ViT3DTower(in_chans=in_chans,img_size=img_size,patch_size=patch_size,
                             select_features=select_features,select_layer=select_layer,
                             spatial_dims=spatial_dims)

        check = torch.load("/home/cyx/Datasets/pretrained_ViT.bin", map_location="cpu")

        for k in list(check.keys()):
            check["vision_tower." + k] = check[k]
            del check[k]

        msg = image_encoder.load_state_dict(check, strict=False)

        for name, values in image_encoder.named_parameters():
            values.requires_grad = False

        for params in image_encoder.vision_tower.after_confusion.parameters():
            params.requires_grad = True

        #  存 lora的参数
        self.w_As = []
        self.w_Bs = []
        self.conv_As = []

        # 增加lora adapter
        for t_layer_i, blk in enumerate(image_encoder.vision_tower.blocks):
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
                                     r=r, conv_lora=ConvMoe)

        self.reset_parameters()

        self.image_encoder = image_encoder

        #
        if text_prompt:
            self.cls = nn.Sequential(
                nn.Linear(768+1024, 512),
                # nn.LeakyReLU(inplace=True),
                nn.Linear(512, num_classes)
            )
        else:
            self.cls = nn.Linear(768, num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))

        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x, text_prompt = None):
        x = self.image_encoder(x)  # B D
        if text_prompt is not None:
            t_and_i = torch.cat([x, text_prompt],dim=-1)
            y = self.cls(t_and_i)
            return y
        y = self.cls(x)
        return y



if __name__ == "__main__":
    # model = ViT3DTower(in_chans=1, patch_size=[16, 8, 8], img_size=128)
    model = M3D_VIT_CLS(in_chans=1, patch_size=[16, 8, 8], img_size=128,ConvMoe=True,text_prompt=False,select_layer=-1)
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
    #
    #
    # t = torch.randn(1,4,128,128,128)
    # text_p = torch.randn(1,1024)
    # print(model(t,text_p).shape)