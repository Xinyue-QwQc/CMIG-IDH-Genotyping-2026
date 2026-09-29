# import sys
# sys.path.append("./")
import os

from models.SAM_Med3D.image_encoder3DV3 import ImageEncoderViT3D
from models.SAM_Med3D.PEFT_module import spatial_Attention, GRN_3D
# from SAM_Med3D.image_encoder3D import ImageEncoderViT3D
# from SAM_Med3D.PEFT_module import spatial_Attention,GRN_3D
from models.Revised_SAM_Med3D.prompt_extracter import Mismatch_extractv1, Mismatch_extracterv2
# from .prompt_extracter import Mismatch_extractv1,Mismatch_extracterv2
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from models.Revised_SAM_Med3D.ConvMoe3D import MoEConv
from visualizer import get_local
from models.Revised_SAM_Med3D.Fusion_module import AlignImageText


# from torchvision.models import resnet50
# This version is text guide version
# Lora 微调代码
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

            qkv[:, :, :, :, self.dim:-self.dim] += new_k

        qkv[:, :, :, :, : self.dim] += new_q
        qkv[:, :, :, :, -self.dim:] += new_v
        return qkv


class _SE_qkv(nn.Module):
    def __init__(self, qkv: nn.Module, conv_a_q: nn.Module, conv_b_q: nn.Module,
                 conv_a_v: nn.Module, conv_b_v: nn.Module) -> None:
        super().__init__()
        self.qkv = qkv
        self.conv_a_q = conv_a_q
        self.conv_b_q = conv_b_q

        self.conv_a_v = conv_a_v
        self.conv_b_v = conv_b_v

        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # B D H W 3*dim  qkv

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        new_q = self.conv_b_q(self.conv_a_q(x))
        new_v = self.conv_b_v(self.conv_a_v(x))

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        new_q = new_q.permute(0, 2, 3, 4, 1).contiguous()
        new_v = new_v.permute(0, 2, 3, 4, 1).contiguous()

        qkv[:, :, :, :, : self.dim] += new_q
        qkv[:, :, :, :, -self.dim:] += new_v

        return qkv


# SE and lora
class _SE_and_lora_qkv(nn.Module):
    def __init__(self, qkv: nn.Module,
                 conv_a_q: nn.Module, conv_a_v: nn.Module,
                 linear_a_q: nn.Module, linear_a_v: nn.Module,
                 linear_b_q: nn.Module, linear_b_v: nn.Module,
                 scale_learnable=True, scale_factor=1.0, weighted_factor=0.5) -> None:
        super().__init__()

        self.qkv = qkv

        self.conv_a_q = conv_a_q
        self.conv_a_v = conv_a_v

        self.linear_a_q = linear_a_q
        self.linear_a_v = linear_a_v

        self.linear_b_q = linear_b_q
        self.linear_b_v = linear_b_v

        self.dim = qkv.in_features

        if scale_learnable:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = scale_factor

        self.weighted_favtor = weighted_factor

    def forward(self, x):

        qkv = self.qkv(x)  # B D H W 3*dim  qkv

        # 获得lora线性层
        linear_a_q_down = self.linear_a_q(x)
        linear_a_v_down = self.linear_a_v(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # 获得分组卷积
        conv_a_q_down = self.conv_a_q(x).permute(0, 2, 3, 4, 1).contiguous()
        conv_a_v_down = self.conv_a_v(x).permute(0, 2, 3, 4, 1).contiguous()

        x = x.permute(0, 2, 3, 4, 1).contiguous()

        q_down = self.weighted_favtor * linear_a_q_down + (1 - self.weighted_favtor) * conv_a_q_down
        v_down = self.weighted_favtor * linear_a_v_down + (1 - self.weighted_favtor) * conv_a_v_down

        new_q = self.linear_b_q(q_down)
        new_v = self.linear_b_v(v_down)

        qkv[:, :, :, :, : self.dim] += self.scale * new_q
        qkv[:, :, :, :, -self.dim:] += self.scale * new_v

        return qkv


# 进行维度变化  只使用一层
class Stem_net(nn.Module):

    def __init__(self, in_channels=4, hidden_channels=128, out_channels=1, activation=nn.LeakyReLU(inplace=True),
                 norm_layer=nn.BatchNorm3d) -> None:
        super().__init__()

        self.conv0 = nn.Conv3d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1)
        self.bn0 = norm_layer(hidden_channels)

        self.conv1 = nn.Conv3d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn1 = norm_layer(out_channels)

        self.act_fn = activation

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act_fn(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        return x

    # 将transformer得到的特征进行处理，最后进行分类


class Cls_net(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm3d, pool_layer=nn.AdaptiveAvgPool3d, input_channels=384,
                 num_classes=2) -> None:
        super().__init__()

        self.bn0 = norm_layer(input_channels)
        self.pool = pool_layer(1)
        self.ly0 = nn.Linear(in_features=input_channels, out_features=num_classes)
    # @get_local("temp")
    def forward(self, x):
        x = self.bn0(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # temp = x
        y = self.ly0(x)
        return y

class Plain_cls_net(nn.Module):
    def __init__(self,in_channels=384,num_classes=2):

        super().__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.drop = nn.Dropout(0.2)

        self.fc = nn.Sequential(
            nn.Linear(in_channels*2,256),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x = self.norm(x)
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        max_pool = max_pool.view(max_pool.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.drop(x)


        y = self.fc(x)

        return y



# Text guide version  原始需要*2
class Text_guied_cls(nn.Module):
    def __init__(self, in_channels_img=384, in_channels_text=1024, out_channels=768, scale=1.0, n=4):
        super().__init__()
        # self.img2text = nn.Linear(in_channels_img, out_channels, bias=False)
        # self.text2img = nn.Linear(in_channels_text, out_channels, bias=False)
        # self.pool = nn.AdaptiveAvgPool3d(1)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.gt_label = nn.Parameter(torch.arange(n), requires_grad=False)

    def forward(self, x_img, text_feature):
        # 处理图像的特征
        # x_img = self.pool(x).reshape(x.shape[0], -1)
        # x_img = self.img2text(x_img)
        # 进行norm之后再计算相似度
        x_img = F.normalize(x_img, dim=-1)

        # 处理文本的特征
        # text_feature = self.text2img(text_feature)
        # text = text_feature
        text_feature = F.normalize(text_feature, dim=-1)

        logits_per_img = x_img @ text_feature.T / self.scale
        logits_per_text = logits_per_img.T

        n = logits_per_text.shape[0]

        loss_img2text = F.cross_entropy(logits_per_img, self.gt_label[:n], ignore_index=-1)
        loss_text2img = F.cross_entropy(logits_per_text, self.gt_label[:n], ignore_index=-1)

        # 平衡一下数量级/10
        return (loss_text2img + loss_img2text).mean() / 10


class Cls_netv2(nn.Module):
    def __init__(self, in_channels=768, hidden_channels1=256, hidden_channels2=32, num_classes=2,
                 text_embed=False) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.norm = nn.BatchNorm3d(in_channels)
        self.drop = nn.Dropout(0.2)

        self.text_embed = text_embed

        if text_embed:
            self.text_proj = nn.Linear(1024, in_channels * 2)

        self.hidden_layer1 = nn.Linear(in_channels * 2, hidden_channels1)
        self.hidden_layer2 = nn.Linear(hidden_channels1, hidden_channels2)
        self.classifier = nn.Linear(hidden_channels2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, text_embeddings: Optional[torch.Tensor] = None):
        x = self.norm(x)
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        max_pool = max_pool.view(max_pool.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.drop(x)

        if text_embeddings is None:
            x = self.hidden_layer1(x)
        else:
            text_embeddings = self.text_proj(text_embeddings)
            x = self.hidden_layer1(x + text_embeddings)

        # x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.classifier(x)
        if text_embeddings is None:
            return x
        else:
            return x, text_embeddings


class CLS_Net_revised(nn.Module):
    def __init__(self,in_channels=768, hidden_channels1=256, hidden_channels2=32, num_classes=2,
                 text_embed=False):
        super().__init__()



        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.norm = nn.BatchNorm3d(in_channels)
        self.drop = nn.Dropout(0.2)

        self.text_embed = text_embed

        if text_embed:
            self.text_proj = nn.Linear(1024, in_channels * 2)
            self.text_guided = Text_guied_cls()

        self.hidden_layer1 = nn.Linear(in_channels * 2, hidden_channels1)
        self.hidden_layer2 = nn.Linear(hidden_channels1, hidden_channels2)
        self.classifier = nn.Linear(hidden_channels2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, text_embeddings: Optional[torch.Tensor] = None):

        x = self.norm(x)

        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        max_pool = max_pool.view(max_pool.size(0), -1)

        x = torch.cat([avg_pool, max_pool], dim=1)

        x = self.drop(x)

        if text_embeddings is None:
            x = self.hidden_layer1(x)
        else:
            text_embeddings = self.text_proj(text_embeddings)

            loss = self.text_guided(x, text_embeddings)

            x = self.hidden_layer1(x + text_embeddings)


        # x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.classifier(x)
        if text_embeddings is None:
            return x
        else:
            return x, loss

class CLS_Net_revisedV2(nn.Module):
    def __init__(self,in_channels=768, hidden_channels1=256, hidden_channels2=32, num_classes=2,
                 text_embed=False):
        super().__init__()



        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.norm = nn.BatchNorm3d(in_channels)
        self.drop = nn.Dropout(0.2)

        self.text_embed = text_embed

        if text_embed:
            self.text_proj = nn.Linear(1024, in_channels * 2)
            self.text_guided = Text_guied_cls()

        self.hidden_layer1 = nn.Linear(in_channels * 2, hidden_channels1)

        self.hidden_layer2 = nn.Linear(hidden_channels1, hidden_channels2)
        self.classifier = nn.Linear(hidden_channels2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, text_embeddings: Optional[torch.Tensor] = None):

        x = self.norm(x)

        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        max_pool = max_pool.view(max_pool.size(0), -1)


        x = torch.cat([avg_pool, max_pool], dim=1)

        x = self.drop(x)

        if text_embeddings is None:
            x = self.hidden_layer1(x)
        else:
            text_embeddings = self.text_proj(text_embeddings)

            loss = self.text_guided(x, text_embeddings)

            x = self.hidden_layer1(x + text_embeddings)


        # x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.classifier(x)
        # return x
        if text_embeddings is None:
            return x
        else:
            return x, loss

class Cls_netv3(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm3d, pool_layer=nn.AdaptiveAvgPool3d, input_channels=384,
                 hidden_channels=128,
                 num_classes=2) -> None:
        super().__init__()

        self.bn0 = norm_layer(input_channels)
        self.pool = pool_layer(1)
        self.cls = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x):
        x = self.bn0(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        y = self.cls(x)
        return y


# Multi Stage Fusion
class CLS_Netv4(nn.Module):

    def __init__(self, in_channels, hidden_channels1=256, hidden_channels2=32, num_classes=2):
        super().__init__()
        self.bn0 = nn.BatchNorm3d(in_channels)
        self.conv0 = nn.Conv3d(in_channels, hidden_channels1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.hidden_layer2 = nn.Linear(hidden_channels1, hidden_channels2)
        self.classifier = nn.Linear(hidden_channels2, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv0(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.hidden_layer2(x)
        x = self.classifier(x)
        return x


class Fusion_cls_net(nn.Module):
    def __init__(self, in_channels=384, hidden_channels=192, out_channels=2, norm_layer=GRN_3D) -> None:
        super().__init__()
        self.norm_layer = norm_layer(mode="Second", dim=in_channels)
        self.spatial_layer = spatial_Attention()
        self.hidden1 = nn.Conv3d(in_channels, hidden_channels, groups=hidden_channels, kernel_size=3, stride=1,
                                 padding=1)
        self.norm = nn.BatchNorm3d(hidden_channels)
        self.hidden1_1 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.hidden2 = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        )

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.spatial_layer(x)
        x = self.hidden1(x)
        x = self.norm(x)
        x = self.hidden1_1(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.hidden2(x)
        y = self.classifier(x)
        return y


class Fusion_cls_netv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, norm_layer=nn.BatchNorm3d) -> None:
        super().__init__()
        self.norm = norm_layer(in_channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class SAM_CLS(nn.Module):

    def __init__(self, check_path="/home/cyx/Datasets/sam_med3d_brain.pth",
                 r=4, after_fusion_channels=128,
                 in_channels=4, conv_adapter=False, Adapter_former=False,
                 parallel=True, mismatch=False, ConvMoe=False, text_embed=False,
                 reduction_ratio=4, num_classes=2,bg_know=False) -> None:
        super().__init__()

        self.mismatch = mismatch
        self.know_embed: Optional[nn.Parameter] = None
        if bg_know:
            # self.know_embed = nn.Parameter(torch.zeros(1,1024))
            self.know_embed = nn.Parameter(torch.load("/home/cyx/Codes/SAM_MED3D_for_CLS/models/Language_model/background.pt"),requires_grad=False) # ,requires_grad=False
        # 导入预训练模型  仅仅使用encoder的
        check = torch.load(check_path, map_location="cpu")
        model_params = check["model_state_dict"]

        # 方便导入 去除前缀 同时去除prompt encoder和decoder的参数
        for k in list(model_params.keys()):

            if k.startswith("image_encoder"):
                model_params[k[len("image_encoder."):]] = model_params[k]

            del model_params[k]

        image_encoder = ImageEncoderViT3D(img_size=128, patch_size=16, out_chans=384,
                                          global_attn_indexes=[2, 5, 8, 11],
                                          use_rel_pos=True, window_size=14,
                                          conv_adapter=conv_adapter, adapt_former=Adapter_former,
                                          after_fusion_channels=after_fusion_channels,
                                          parallel=parallel, Mis_branch=mismatch)

        msg = image_encoder.load_state_dict(model_params, strict=False)


        # 冻结所有参数
        for name, values in image_encoder.named_parameters():
            values.requires_grad = False

        for params in image_encoder.after_confusion.parameters():
            params.requires_grad = True

        # for params in image_encoder.neck_adapter.parameters():
        #     params.requires_grad = True

        if mismatch:
            for params in image_encoder.mis_extractor.parameters():
                params.requires_grad = True
            for i in image_encoder.blocks:
                for name, values in i.named_parameters():
                    if "mis_attention" in name:
                        # print(name)
                        values.requires_grad = True
                    if "proj_mismatch" in name:
                        values.requires_grad = True
            # print("after_confusion params shape is {}".format(params.shape))


        if conv_adapter:
            count = -1
            for i in image_encoder.blocks:
                count += 1
                if count not in [2,5,8,11]:
                    # for params in i.conv_adapter0.parameters():
                    #     params.requires_grad = True
                    for params in i.conv_adapter.parameters():
                        params.requires_grad = True
        if Adapter_former:
            for i in image_encoder.blocks:
                for params in i.adapter_former.parameters():
                    params.requires_grad = True


        #  存 lora的参数
        self.w_As = []
        self.w_Bs = []
        self.conv_As = []

        # 增加lora adapter
        for t_layer_i, blk in enumerate(image_encoder.blocks):
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

        # self.fusion_align = AlignImageText(img_channels=self.dim,text_channels=1024,reduction_ratio=reduction_ratio)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.dim//reduction_ratio,32),
        #     nn.Linear(32,num_classes)
        # )


        # self.x_proj = nn.Conv3d(384,768,kernel_size=1,bias=False)


        # self.cls_net = Plain_cls_net()        #
        self.cls_net = Cls_net()

        # self.text_fc = nn.Linear(2,2,bias=False)

        # self.img_text = Text_guied_cls()
        # self.cls_net = CLS_Netv4(in_channels=768)
        # self.cls_net = CLS_Net_revisedV2(text_embed=text_embed, in_channels=768)
        # self.cls_net = Fusion_cls_net()
        # self.cls_net = Fusion_cls_netv2(in_channels=384,hidden_channels=128)
        # self.temp = nn.Parameter(torch.ones(1))
        # self.alpha = nn.Parameter(torch.ones(1))

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))

        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
        # for w_A in self.w_As:
        #     nn.init.zeros_(w_A.weight)
        #
        # for w_B in self.w_Bs:
        #     nn.init.kaiming_uniform_(w_B.weight, a=math.sqrt(5))


    # @get_local("temp")
    def forward(self, x, text_embeddings: Optional[torch.Tensor] = None):  #text_embeddings: Optional[torch.Tensor] = None, global_text: Optional[torch.Tensor] = None

        # if self.know_embed is not None:
        #     text_embeddings = self.know_embed
        if text_embeddings is not None:
            if self.know_embed is not None:
                text_embeddings = text_embeddings + self.know_embed
            # text_embeddings = self.text2img(text_embeddings)
            x = self.image_encoder(x,text_embeddings)
        else:
            x = self.image_encoder(x)
        # temp = x
        y = self.cls_net(x)
        return y


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model = SAM_CLS(conv_adapter=True, ConvMoe=True, mismatch=True, text_embed=True,bg_know=True)

    print(model)

    total = 0
    all_params = 0
    for name, values in model.named_parameters():
        if values.requires_grad:
            print(name,values.shape)
            total += values.nelement()
        all_params += values.nelement()
    print(total / 1e6)
    print(total * 100 / all_params)
    print("ALL", all_params / 1e6)

    # check = torch.load("/home/cyx/Datasets/sam_med3d.pth",map_location="cpu")
    # model_parms = check["model_state_dict"]

    # model.load_state_dict(model_parms,strict=False)

    # for m in t:
    #     print("\tStart\t")
    #     for d in m:
    #         print(d)
    # pass
    # 初始化实例化网络
    # stem = stem_net()
    # cls = cls_net()
    # image_encoder = ImageEncoderViT3D(img_size=128,patch_size=16,out_chans=384,global_attn_indexes=[2, 5, 8, 11],use_rel_pos=True)

    # # # 得到分类网络
    # model = SAM_CLS(image_encoder=image_encoder,cls_net=cls,stem_net=stem)

    t = torch.randn(4, 4, 128, 128, 128)
    # text_embed = torch.randn(4, 32, 1024)
    t_g = torch.randn(4,1024)  #,text_embed,t_g
    # # # # #
    print(model(t,t_g).shape)

