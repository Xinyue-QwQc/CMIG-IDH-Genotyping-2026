# import sys
# sys.path.append("./")
from models.SAM.image_encoder import ImageEncoderViT
# from models.SAM_Med3D.image_encoder3D import ImageEncoderViT3D
# from models.SAM_Med3D.PEFT_module import spatial_Attention,GRN_3D

# from models.Revised_SAM_Med3D.prompt_extracter import Mismatch_extractv1,Mismatch_extracterv2

import torch
import torch.nn as nn
import math 
from typing import Optional, Tuple
import torch.nn.functional as F 
# from torchvision.models import resnet50

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
     

    def forward(self, x):
        qkv = self.qkv(x)  # B D H W 3*dim  qkv 
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        if self.linear_a_k is not None:
            new_k = self.linear_b_k(self.linear_a_k(x))
            qkv[:, :, :,  self.dim:-self.dim] += new_k
        qkv[:, :, :, : self.dim] += new_q
        qkv[:,  :, :,  -self.dim:] += new_v 
        return qkv


class _SE_qkv(nn.Module):
    def __init__(self,qkv: nn.Module,conv_a_q:nn.Module,conv_b_q:nn.Module,
                 conv_a_v:nn.Module,conv_b_v:nn.Module) -> None:
        super().__init__()
        self.qkv = qkv
        self.conv_a_q = conv_a_q
        self.conv_b_q = conv_b_q 

        self.conv_a_v = conv_a_v
        self.conv_b_v = conv_b_v 
        
        self.dim = qkv.in_features
     

    def forward(self, x):
        qkv = self.qkv(x)  # B D H W 3*dim  qkv 

        x = x.permute(0,4,1,2,3).contiguous()

        new_q = self.conv_b_q(self.conv_a_q(x))
        new_v = self.conv_b_v(self.conv_a_v(x))

        x = x.permute(0,2,3,4,1).contiguous()
        new_q = new_q.permute(0,2,3,4,1).contiguous()
        new_v = new_v.permute(0,2,3,4,1).contiguous()

        qkv[:, :, :, :, : self.dim] += new_q
        qkv[:, :, :, :,  -self.dim:] += new_v 

        return qkv

# SE and lora
class _SE_and_lora_qkv(nn.Module):
    def __init__(self,qkv: nn.Module,
                 conv_a_q:nn.Module,conv_a_v:nn.Module,
                 linear_a_q: nn.Module,linear_a_v: nn.Module,
                 linear_b_q:nn.Module,linear_b_v:nn.Module,
                 scale_learnable=True,scale_factor=1.0,weighted_factor=0.5) -> None:
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

        x = x.permute(0,4,1,2,3).contiguous()

        # 获得分组卷积
        conv_a_q_down = self.conv_a_q(x).permute(0,2,3,4,1).contiguous()
        conv_a_v_down = self.conv_a_v(x).permute(0,2,3,4,1).contiguous()

        x = x.permute(0,2,3,4,1).contiguous()

        q_down = self.weighted_favtor*linear_a_q_down + (1-self.weighted_favtor)*conv_a_q_down
        v_down = self.weighted_favtor*linear_a_v_down + (1-self.weighted_favtor)*conv_a_v_down

        new_q = self.linear_b_q(q_down)
        new_v = self.linear_b_v(v_down)


        qkv[:, :, :, :, : self.dim] += self.scale*new_q
        qkv[:, :, :, :,  -self.dim:] += self.scale*new_v 

        return qkv


# 进行维度变化  只使用一层
class Stem_net(nn.Module):

    def __init__(self,in_channels=4,hidden_channels=128,out_channels=1,
                 activation=nn.LeakyReLU(inplace=True),norm_layer=nn.BatchNorm3d,depth_k=1,depth_s=1) -> None:
        super().__init__()

        self.conv0 = nn.Conv3d(in_channels=in_channels,out_channels=hidden_channels,kernel_size=(depth_k,1,1),stride=(depth_s,1,1))
        self.bn0 = norm_layer(hidden_channels)

        self.conv1 = nn.Conv3d(hidden_channels,hidden_channels,kernel_size=(depth_k,1,1),stride=(depth_s,1,1))
        self.bn1 = norm_layer(hidden_channels)

        self.conv2 = nn.Conv3d(in_channels=hidden_channels,out_channels=out_channels,kernel_size=(depth_k,1,1),stride=(depth_s,1,1))
        self.bn2 = norm_layer(out_channels)
        
        self.act_fn = activation

        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        )
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act_fn(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_fn(x)

        return x 

# 将transformer得到的特征进行处理，最后进行分类
class Cls_net(nn.Module):

    def __init__(self,norm_layer=nn.BatchNorm3d,pool_layer=nn.AdaptiveAvgPool3d,input_channels=256,num_classes=2) -> None:
        super().__init__()

        self.bn0 = norm_layer(input_channels)
        self.pool = pool_layer(1)
        self.ly0 = nn.Sequential(
                                  nn.Linear(input_channels, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, num_classes))
    
    def forward(self,x):
        x = self.bn0(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        y = self.ly0(x)
        return y 


def resized(rel_pos,max_rel_dist):
    rel_pos_resized = F.interpolate(
    rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
    size=max_rel_dist,
    mode="linear",)
    rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    return rel_pos_resized



class LVM_SAM_CLS(nn.Module):

    def __init__(self,check_path="/home/cyx/Datasets/lvmmed_vit.pth",num_slices=128,
                 r=4,after_fusion_channels=128,depth_adapter= True,depth_hiden_channels=128,
                 in_channels=4,conv_adapter=False,Adapter_former=False,parallel=True) -> None:
        super().__init__()

        # 导入预训练模型  仅仅使用encoder的
        check = torch.load(check_path,map_location="cpu")

        image_encoder = ImageEncoderViT(window_size=14,global_attn_indexes=[2,5,8,11],use_rel_pos=True,
                                        use_abs_pos=False,img_size=128,num_slices=num_slices,
                                        depth_hiden_channels=depth_hiden_channels
                                        )
        # 为方便导入，对相对位置编码进行插值
        for i in [2,5,8,11]:
            rel_pos_h = f"blocks.{i}.attn.rel_pos_h"
            rel_pos_w = f"blocks.{i}.attn.rel_pos_w"
            check[rel_pos_h] = resized(check[rel_pos_h],15)
            check[rel_pos_w] = resized(check[rel_pos_w],15)

        
        msg = image_encoder.load_state_dict(check,strict=False)

        # print(msg)

        # 冻结所有参数
        for p in image_encoder.parameters():
            p.requires_grad = False

        if num_slices > 1:
            for p in image_encoder.patch_embed.depth_embed.parameters():
                p.requires_grad = True 

        if depth_adapter:
            for i in image_encoder.blocks:
                for params in i.depth_adapter0.parameters():
                    params.requires_grad = True 
                for params in i.depth_adapter1.parameters():
                    params.requires_grad = True 
        # for params in image_encoder.after_confusion.parameters():
        #     params.requires_grad = True
            # print("after_confusion params shape is {}".format(params.shape))
        
        if conv_adapter:
            for i in image_encoder.blocks:
                # for params in i.conv_adapter0.parameters():
                #     params.requires_grad = True
                for params in i.conv_adapter.parameters():
                    params.requires_grad = True
        if Adapter_former:
            for i in image_encoder.blocks:
                for params in i.adapter_former.parameters():
                    params.requires_grad = True
                    
        # for i in image_encoder.blocks:
        #     for params in i.attn.moe_lora.parameters():
        #         params.requires_grad = True

      
        
        
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
        #     # w_a_linear_k = nn.Linear(self.dim, r, bias=False)
        #     # w_b_linear_k = nn.Linear(r, self.dim, bias=False)
        # #     # w_a_conv_q = nn.Conv3d(in_channels=self.dim,out_channels=r,stride=1,kernel_size=3,padding=1,groups=r,bias=False)
        # #     # w_b_linear_q = nn.Conv3d(in_channels=r,out_channels=self.dim,kernel_size=1,bias=False)
        # #     # w_a_conv_v = nn.Conv3d(in_channels=self.dim,out_channels=r,stride=1,kernel_size=3,padding=1,groups=r,bias=False)
        # #     # w_b_linear_v = nn.Conv3d(in_channels=r,out_channels=self.dim,kernel_size=1,bias=False)
            
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

        #     self.w_As.append(w_a_linear_k)
        #     self.w_Bs.append(w_b_linear_k)

        #     # self.conv_As.append(w_a_conv_q)
        #     # self.conv_As.append(w_a_conv_v)

        #     # blk.attn.qkv = _SE_and_lora_qkv(w_qkv_linear,
        #     #                 w_a_conv_q,w_a_conv_v,
        #     #                 w_a_linear_q,w_a_linear_v,
        #     #                 w_b_linear_q,w_b_linear_v)

            blk.attn.qkv = _LoRA_qkv(w_qkv_linear,
                                    w_a_linear_q,w_a_linear_v,
                                    w_b_linear_q,w_b_linear_v,
                                   ) #  w_a_linear_k,w_b_linear_k

        #     # blk.attn.qkv = _SE_qkv(
        #     #     w_qkv_linear,
        #     #     w_a_conv_q,
        #     #     w_b_linear_q,
        #     #     w_a_conv_v,
        #     #     w_b_linear_v,
        #     # )

        # self.reset_parameters()

        self.stem = Stem_net(in_channels=in_channels,hidden_channels=256,out_channels=3,depth_k=1,depth_s=1)
        self.image_encoder = image_encoder
        # self.mismatch_extracter = Mismatch_extracterv2(dims=[16,32,64,128],final_channles=384)
        # self.stem = Stem_net()
        self.cls_net = Cls_net() 
        # self.cls_net = Cls_netv2()
        # self.cls_net = Fusion_cls_net()
        # self.cls_net = Fusion_cls_net(in_channels=768,hidden_channels=256)
        # self.temp = nn.Parameter(torch.ones(1))
    
    def reset_parameters(self) -> None:
        # for conv_A in self.conv_As:
        #      nn.init.kaiming_normal_(conv_A.weight,
        #                                 mode='fan_out',
        #                                 nonlinearity='relu')
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))

        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
  
       
    
    def forward(self,x):
        
        # B 1 128 128 128
        x = self.stem(x)

        B,C,D,H,W = x.shape
        x = x.permute(0,2,1,3,4)
        x = x.contiguous().view(B*D,C,H,W)

        # B 256 8 8 8 
        # 提取T2-Flair信号
        # mis_signs = self.mismatch_extracter(x[:,-2,...].unsqueeze(1),x[:,-1,...].unsqueeze(1))
        # image_embed = self.image_encoder(x)
        x = self.image_encoder(x)

        B_,C_,H_,W_ = x.shape

        x = x.view(B_//8,8,C_,H_,W_)
        x = x.permute(0,2,1,3,4)

        # x = torch.cat([image_embed,mis_signs],dim=1)

        # B 2
        y = self.cls_net(x)
        # return x,ls

        # return y/self.temp
        return y 


if __name__ == "__main__":
    model = LVM_SAM_CLS(conv_adapter=False)

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

    t = torch.randn(4,4,128,128,128)

    print(model(t).shape)

