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
from models.SAM_Med3D.PEFT_module import GITCrossAdapterMoEPool, MisBlock, GN_Mis, GroupSCA, GroupSCA_V2
import math
# from models.baseline.transformerblock import TransformerBlock
# from models.baseline.ConvMoe import MoEConv
# from .ConvMoe import MoEConv
from typing import Optional


class After_fusion(nn.Module):
    def __init__(self, in_chans=768 * 4, hidden_chans=128, out_chans=768, activation=nn.LeakyReLU(inplace=True),
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.down = nn.Linear(in_chans, hidden_chans)
        self.norm0 = norm_layer(hidden_chans)

        self.up = nn.Linear(hidden_chans, out_chans)
        self.norm1 = norm_layer(out_chans)

        self.act_fn = activation

        # 初始化参数的方式
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # elif isinstance(m,nn.LayerNorm):

    def forward(self, x):

        x = self.down(x)
        x = self.norm0(x)
        x = self.act_fn(x)

        x = self.up(x)
        x = self.norm1(x)
        x = self.act_fn(x)

        return x


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
            use_adapter=False,
            use_mis=False
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
        self.use_adapter = use_adapter
        self.use_mis = use_mis
        self.num_layers = num_layers

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )

        if use_adapter:
            self.adapters = nn.ModuleList(
                [
                    GITCrossAdapterMoEPool(in_channels=hidden_size)
                    # MoEPoolCrossAttn(in_channels=hidden_size)
                    for i in range(num_layers)
                ]
            )
        else:
            self.adapters = nn.ModuleList(
                [
                    nn.Identity()
                    for i in range(num_layers)
                ]
            )

        num = 4

        if use_mis:
            self.extract_mis_adapter = nn.ModuleList(
                [
                    MisBlock(in_channels=hidden_size, hidden_channels=hidden_size // 16)  # original 1/16
                    for i in range(num)
                ]
            )
            self.gn_mis = nn.ModuleList(
                [
                    # GN_Mis(channels=hidden_size)
                    GroupSCA_V2(channels=hidden_size)
                    for i in range(num)
                ]
            )
        else:
            self.extract_mis_adapter = nn.ModuleList(
                [
                    nn.Identity()
                    for i in range(num)
                ]
            )
            self.gn_mis = nn.ModuleList(
                [
                    nn.Identity()
                    for i in range(num)
                ]
            )

        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, text: Optional[torch.Tensor] = None):
        ls = []

        for i in range(4):
            ls.append(self.patch_embedding(x[:, i, :, :, :].unsqueeze(1)))  # B 2048 768

        t2, flair = ls[-2], ls[-1]

        x_ = torch.cat(ls, dim=-1)
        x = self.after_confusion(x_)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []

        for i in range(self.num_layers):
            if self.use_adapter:
                if self.use_mis and i % 3 == 2:  # i%2==1
                    mis_index = (i - 2) // 3  # 1 // 2    (i-2)//3
                    mis_sign, t2, flair = self.extract_mis_adapter[mis_index](t2, flair)  # mis_index
                    # mis_score = self.gn_mis[mis_index](mis_sign)
                    # x[:,1:,:] += mis_score  # 变成+
                    x[:, 1:, :] = self.gn_mis[mis_index](mis_sign, x[:, 1:, :])

                x_new = self.adapters[i](x, text)
                x = self.blocks[i](x)
                x = x + x_new
            else:
                # if self.use_adapter:
                #     x = self.blocks[i](x,text)
                # else:
                x = self.blocks[i](x)

            hidden_states_out.append(x)

        x = self.norm(x)
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])
        return x, hidden_states_out


class ViT3DTower(nn.Module):
    def __init__(self, in_chans, img_size, patch_size, select_features="patch", select_layer=-1,
                 spatial_dims=3, use_adapter=False, use_mis=False):
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
            use_adapter=use_adapter,
            use_mis=use_mis
        )

    def forward(self, images, text: Optional[torch.Tensor] = None):
        last_feature, hidden_states = self.vision_tower(images, text)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer == 1:
            image_features = torch.cat([hidden_states[2], last_feature], dim=-1)
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            # 对结果取mean
            image_features = torch.mean(image_features[:, 1:], dim=1, keepdim=False)
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


class TextFeatureAggregator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(in_channels)
        self.proj_mis = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )
        # self.norm2 = nn.LayerNorm(out_channels)

        # self.conv = nn.Conv1d(2, 1, 3, 1, 1)   # feature reduction

    def forward(self, text_features, modality_num=4) -> torch.Tensor:
        assert len(text_features) % modality_num == 0  # 各个模态能够被整除

        text_features = text_features.view(-1, modality_num, text_features.size(-1))  # t1 t1ce t2 flair
        # fix bug
        # text_features = text_features.permute(1,0,2).contiguous()

        text_features = self.norm1(text_features)
        text_features_max, _ = torch.max(text_features, dim=1)
        text_features_avg = torch.mean(text_features, dim=1)
        # text_features = text_features_avg
        text_features = torch.cat([text_features_avg, text_features_max], dim=-1)
        text_features = self.proj_mis(text_features)
        # text_features = self.norm2(text_features)

        # 设想是 t1 t1ce一同进行处理，然后t2-flair另外进行处理
        return text_features


class TextFeatureAggregatorV2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(in_channels)
        # self.conv = nn.Conv1d(4, 1, 7, 1, 3)   # feature reduction

        self.proj_mis = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, text_features, modality_num=4) -> torch.Tensor:
        assert len(text_features) % modality_num == 0  # 各个模态能够被整除

        text_features = text_features.view(-1, modality_num, text_features.size(-1))  # t1 t1ce t2 flair

        text_features = self.norm1(text_features)

        max_text, _ = torch.max(text_features, dim=1)
        # min_text,_ = torch.min(text_features,dim=1,keepdim=True)   #min_text,
        avg_text = torch.mean(text_features, dim=1)
        text_features = torch.cat([max_text, avg_text], dim=1)
        # text_features = self.conv(text_features)[:,0]

        text_features = self.proj_mis(text_features)

        text_features = self.norm2(text_features)

        # 设想是 t1 t1ce一同进行处理，然后t2-flair另外进行处理
        return text_features


class M3D_VIT_CLS(nn.Module):
    def __init__(self, in_chans, img_size, patch_size, select_features="cls_patch", select_layer=-1,
                 spatial_dims=3, num_classes=2, use_adapter=False, use_mis=False,with_bg=False):
        super().__init__()

        image_encoder = ViT3DTower(in_chans=in_chans, img_size=img_size, patch_size=patch_size,
                                   select_features=select_features, select_layer=select_layer,
                                   spatial_dims=spatial_dims, use_adapter=use_adapter,
                                   use_mis=use_mis)

        check = torch.load("/home/cyx/Datasets/pretrained_ViT.bin", map_location="cpu")

        for k in list(check.keys()):
            check["vision_tower." + k] = check[k]
            del check[k]
        
        if with_bg:
            self.bg = nn.Parameter(torch.load("/home/cyx/Codes/SAM_MED3D_for_CLS/BG_PT.pt"),requires_grad=False)
        else:
            self.bg = None 

        msg = image_encoder.load_state_dict(check, strict=False)

        for name, values in image_encoder.named_parameters():
            values.requires_grad = False

        for params in image_encoder.vision_tower.after_confusion.parameters():
            params.requires_grad = True

        if use_adapter:
            for params in image_encoder.vision_tower.adapters.parameters():
                params.requires_grad = True

        if use_mis:
            for params in image_encoder.vision_tower.extract_mis_adapter.parameters():
                params.requires_grad = True
            for params in image_encoder.vision_tower.gn_mis.parameters():
                params.requires_grad = True
            # for k,v in image_encoder.named_parameters():
            #     if "adapter" in k:
            #         v.requires_grad = True

        self.image_encoder = image_encoder

        # self.text_aggregator = TextFeatureAggregator(in_channels=768, hidden_channels=64, out_channels=768)
        # self.text_aggregator = TextFeatureAggregatorV2(in_channels=768, hidden_channels=128, out_channels=768)

        # self.cls = nn.Linear(768, num_classes)
        self.cls = nn.Sequential(
            nn.Linear(768,32),
            nn.Linear(32,2)
        )

    def forward(self, x, text_prompt=None,patient_info = None):
        if text_prompt is not None:
            # text_prompt = text_prompt.view(-1, 4, text_prompt.size(-1))
            # text_prompt = self.text_aggregator(text_prompt)
            if self.bg is not None:
                text_prompt += self.bg
            x = self.image_encoder(x, text_prompt)
        else:
            x = self.image_encoder(x)  # B D
        # if patient_info is not None:
        #     y = self.cls(x+patient_info) #
        # else:
        y = self.cls(x)
        return y


if __name__ == "__main__":
    model = M3D_VIT_CLS(in_chans=1, patch_size=[16, 8, 8], img_size=128, select_layer=-1,
                        use_adapter=True, use_mis=True)

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
    t = torch.randn(4, 4, 128, 128, 128)
    text_prompt = torch.randn(4, 768)
    # text_p = torch.randn(1, 768)
    print(model(t, text_prompt).shape)