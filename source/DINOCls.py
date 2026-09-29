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
# from models.baseline.ConvMoeV2 import MoEConv
# from .ConvMoe import MoEConv
from typing import Optional
from visualizer import get_local

import sys
# sys.path.append('../')  # adjust this to your local path
from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config_3d
from models.baseline.ConvMoeV2 import MoEConv
from models.SAM_Med3D.PEFT_module import ScConv
from models.SAM_Med3D.mismatch_branch import Mismatch_extractorV2
from models.SAM_Med3D.image_encoder3DV3 import Mis_sign_getterV2,Project_layer
from einops import  rearrange


class GenerateMIS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = [160, 208, 296, 244]  # original used
        self.mis_extractor = Mismatch_extractorV2()
        self.mis_attn_scores = nn.ModuleList(
            [Mis_sign_getterV2(in_channels=channels[i],embed_dim=1024) for i in range(4)] 
        )
    
    def forward(self,x):
        T2_Flair_signs = self.mis_extractor(x[:,-2,...].unsqueeze(1),x[:,-1,...].unsqueeze(1))
        mis_scores = [self.mis_attn_scores[i](T2_Flair_signs[i]).squeeze(dim=(1,2,3)).unsqueeze(1) for i in range(4)]
        return mis_scores

class FusionMIS(nn.Module):
    def __init__(self,dim=1024) -> None:
        super().__init__()
        self.pr_layer = Project_layer(in_channels=dim,hidden_channels=128,out_channels=dim,
                                               act_fn="gelu",norm_layer=nn.LayerNorm)
    
    def forward(self,x,scores):
        x[:,1:,:]*=scores
        res = x 
        x = self.pr_layer(x)
        return res + x 



class GITCrossAdapter(nn.Module):
    """
        进行图文融合对齐,并充当adapter的作用进行微调
        in_channels: 输入到模型中的微调参数
        reduction_ratio: 维度缩减的范围
        """

    def __init__(self, in_channels, text_in_channels=1024, reduction_ratio=32) -> None:
        super().__init__()
        hidden_state = in_channels // reduction_ratio

        self.norm = nn.LayerNorm(2 * hidden_state)  #
        self.down = nn.Linear(in_channels, hidden_state)
        self.act = nn.GELU()

        self.text_prj = nn.Linear(text_in_channels,  2 * hidden_state, bias=False)  #
        self.conv_dw = nn.Conv3d(hidden_state, hidden_state, kernel_size=3, stride=1, padding=1, groups=hidden_state)

        self.scconv = ScConv(hidden_state)

        self.up = nn.Linear(hidden_state, in_channels)

    def forward(self, img: torch.Tensor, text):


        # 对文本进行处理
        text = self.text_prj(text)
        text = self.norm(text)
        scale, shift = text.chunk(2, dim=-1)

        # raw_img = img + raw_img
        img = self.down(img)
        img = self.act(img)

        cls = img[:,0,:].unsqueeze(1)
        img = img[:,1:,:].view(img.shape[0],8,8,8,img.shape[-1]).contiguous()

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

        img = rearrange(img, "B C H W D -> B (H W D) C").contiguous()
        img = torch.cat([img,cls],dim=1)
        img = self.up(img)
        return img

class TGFT(nn.Module):
    def __init__(self, mlp:nn.Module,dim=1024) -> None:
        super().__init__()
        self.mlp = mlp
        self.tiff_adapter = GITCrossAdapter(in_channels=dim)

        self.dynamic_param: torch.Tensor
    
    def forward(self,x):
        x_mlp =self.mlp(x)
        x_hybrid = self.tiff_adapter(x,self.dynamic_param)
        return x_hybrid + x_mlp

class MGFT(nn.Module):
    def __init__(self,attn:nn.Module,dim=1024) -> None:
        super().__init__()
        self.attn = attn
        self.fusion = FusionMIS()

        self.dynamic_param: torch.Tensor

    
    def forward(self,x):
        x_attn = self.attn(x) 
        x_fused = self.fusion(x_attn,self.dynamic_param)
        return x_fused


class After_fusion(nn.Module):
    def __init__(self,in_chans=1024*4, hidden_chans=512, out_chans=1024, activation=nn.LeakyReLU(inplace=True),norm_layer=nn.BatchNorm1d):
        super().__init__()

        self.down = nn.Linear(in_chans,hidden_chans)
        # self.norm0 = norm_layer(hidden_chans)

        self.up = nn.Linear(hidden_chans, out_chans)
        # self.norm1 = norm_layer(out_chans)

        self.act_fn = activation

        # 初始化参数的方式

    def forward(self,x):

        x = self.down(x)
        # x = self.norm0(x)
        x = self.act_fn(x)

        x = self.up(x)
        # x = self.norm1(x)
        x = self.act_fn(x)

        return x

def process(x):
    # window_d = ceil(math.pow(window_num,1/3))
    BW,W_num,dim = x.shape

    x = x.view(BW,7,7,7,dim)
    # x = x.permute(0,4,1,2,3).contiguous()
    # x = x.view(BW//window_num,window_d*7,window_d*7,window_d*7,dim)
    return x 

def reverse(x):
    # window_d = ceil(math.pow(window_num,1/3))
    B,D,H,W,C = x.shape
    # x = x.view(B,D//window_d,window_d,H//window_d,window_d,W//window_d,window_d,C)
    # x = x.permute(0,1,3,5,2,4,6,7).contiguous()
    x = x.view(B,-1,C)
    return x 

def process(ln,x):
    x = ln(x)
    return x[:,0,:].unsqueeze(1),x[:,1:,:].view(x.shape[0],8,8,8,x.shape[-1]).contiguous()

def reverse(ln,x,cls):
    x = x.view(x.shape[0],-1,x.shape[-1])
    x = torch.cat([cls,x],dim=1)
    return ln(x)



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
        qkv = self.qkv(x)  # B*window_num window_size dim  x 2 513 3072
        
        if self.conv_lora:
        # 存在cls token需要另行处理
            # x = process_window(x,self.window_num)
            cls_q,new_q = process(self.linear_a_q,x)
            cls_v,new_v = process(self.linear_a_v,x)

            new_q = self.conv_q(new_q)
            new_v = self.conv_v(new_v)

            new_q = reverse(self.linear_b_q,new_q,cls_q)
            new_v = reverse(self.linear_b_v,new_v,cls_v)

            # new_q = reverse_window(new_q,self.window_num)
            # new_v = reverse_window(new_v,self.window_num)
        else:
            new_q = self.linear_b_q(self.linear_a_q(x))
            new_v = self.linear_b_v(self.linear_a_v(x))

        if self.linear_a_k is not None:
            if self.conv_lora:
                cls_k,new_k = process(self.linear_a_k,x)
                new_k = self.conv_k(new_k)
                new_k = reverse(self.linear_b_k,new_k,cls_k)
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
        x_ = torch.cat(ls,dim=-1)
        x = self.after_fusion(x_)
        return  x

class ClsNet(nn.Module):
    def __init__(self,in_dim=1024,hidden_dim=512,classses=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_dim,out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim,out_features=classses)
        )
    def forward(self,x):
        return self.fc(x)    


class DINOCls(nn.Module):
    def __init__(self, num_classes=2, r=4, ConvMoe=False, know=False, multimodal_finetune=False):
        super().__init__()

        config_file = 'train/vit3d_highres'
        pretrained_weights = "/home/cyx/Datasets/HuggingFace_models/DINOV2_3D/3dino_vit_weights.pth"  # adjust this to local path

        self.multimodal_finetune = multimodal_finetune

        if know:
            self.know_embed = nn.Parameter(torch.load("/home/cyx/Codes/SAM_MED3D_for_CLS/models/Language_model/background.pt"),requires_grad=False) # ,requires_grad=False

        cfg = load_and_merge_config_3d(config_file)
        image_encoder = build_model_for_eval(cfg, pretrained_weights)

        for p in image_encoder.parameters():
            p.requires_grad = False

        image_encoder.patch_embed = MultiModalPatchEmbed(image_encoder.patch_embed,After_fusion())

        #  存 lora的参数
        self.w_As = []
        self.w_Bs = []
        self.conv_As = []

        self.mismatch_blocks = [5,11,17,23] # 每个blocks的最后一个block
        # num_windows = [1000,125,27,8]
        # raw_shape = [64,32,16,8]

        for i,block in enumerate(image_encoder.blocks):
            for j,tensorblock in enumerate(block):
                if isinstance(tensorblock,nn.Identity):
                    continue
                w_qkv_linear = tensorblock.attn.qkv
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
                # 5,11,17,23
                tensorblock.attn.qkv = _LoRA_qkv(w_qkv_linear,
                                         w_a_linear_q, w_a_linear_v,
                                         w_b_linear_q, w_b_linear_v,
                                         w_a_linear_k, w_b_linear_k,
                                         r=r,conv_lora=ConvMoe)
                if multimodal_finetune:
                    # mismatch guided finetune
                    if j == self.mismatch_blocks[i]:
                        tensorblock.attn = MGFT(tensorblock.attn) 
                    else:
                        tensorblock.attn = TGFT(tensorblock.attn) 

        self.reset_parameters()

        self.image_encoder = image_encoder

        self.cls_net = ClsNet()

        if self.multimodal_finetune:
             self.generate_mis = GenerateMIS()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))

        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
    # @get_local("temp")
    def forward(self, x, text_embeddings: Optional[torch.Tensor] = None):
        
        # 预先准备好所有外部模态参数
        if self.multimodal_finetune:
            mis_scores = self.generate_mis(x)
            if self.know_embed is not None:
                text_embeddings += self.know_embed
            for i,block in enumerate(self.image_encoder.blocks):
                for j,tensorblock in enumerate(block):
                    if isinstance(tensorblock,nn.Identity):
                        continue
                    if j == self.mismatch_blocks[i]:
                        tensorblock.attn.dynamic_param = mis_scores[i] 
                    else:
                        tensorblock.attn.dynamic_param = text_embeddings
        
        # x_size = x.size(0)
        x = self.image_encoder(x)  # B D  最后返回的是cls token
        # temp = x 
        # return temp
        y = self.cls_net(x)
        # x = x[-1]
        # x_pooled = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x_size, -1)
        return y


if __name__ == "__main__":
    # model =  DINOCls(multimodal_finetune=True,know=True,ConvMoe=True)
    model =  DINOCls(r=24,ConvMoe=True)
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
    # t = torch.randn(1,4,128,128,128)
    # t_g = torch.randn(1,1024)  #,text_embed,t_g

    # print(model(t,t_g).shape)