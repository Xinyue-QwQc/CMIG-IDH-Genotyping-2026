import torch
import torch.nn as nn
from typing import Optional
import math

class Project_layer(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,bias=True,reset_params=False,act_fn:Optional[str]=None,
                 norm_layer:Optional[nn.LayerNorm]=None):

        super().__init__()
        if act_fn is not None:
            if act_fn == "gelu":
                act_layer = nn.GELU()
            elif act_fn == "selu":
                act_layer = nn.SELU(inplace=True)
            elif act_fn == "relu":
                act_layer = nn.LeakyReLU(inplace=True)
        else:
            act_layer = nn.Identity()

        if norm_layer is None:
            norm_layer = nn.Identity()
        else:
            norm_layer = norm_layer(in_channels)

        self.pr_layers = nn.Sequential(
            norm_layer,
            nn.Linear(in_channels,hidden_channels,bias=bias),
            act_layer,
            nn.Linear(hidden_channels,out_channels)
        )

        if reset_params:
            self.reset_parameters()

    def forward(self,x):
        return self.pr_layers(x)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.pr_layers[1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.pr_layers[-1].weight)


class After_confusion(nn.Module):
    
    def __init__(self,in_channels=768,hidden_channels=128,out_channels=768,
                 modify_chans=4,
                 activation=nn.LeakyReLU(inplace=True),norm_layer=nn.BatchNorm3d) -> None:
        super().__init__()

        in_channels *= modify_chans
        self.conv0 = nn.Conv3d(in_channels=in_channels,out_channels=hidden_channels,kernel_size=1,stride=1)
        self.bn0 = norm_layer(hidden_channels)

        self.conv1 = nn.Conv3d(in_channels=hidden_channels,out_channels=out_channels,kernel_size=1,stride=1)
        self.bn1 = norm_layer(out_channels)
        
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

        x = x.permute(0,4,1,2,3).contiguous()
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act_fn(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        x = x.permute(0,2,3,4,1).contiguous()

        return x 

    
# Fact微调具体实现  tt tk pattern
class _Fact_tt(nn.Module):

    def __init__(self,
                 qkv:nn.Module,
                 q_Facts:nn.Module,
                 v_Facts:nn.Module,
                ) -> None:

        super().__init__()

        self.qkv = qkv 

        self.q_Facts = q_Facts
        self.v_Facts = v_Facts


        self.dp_q = nn.Dropout(0.1)
        self.dp_v = nn.Dropout(0.1)
    
    def forward(self,x,FacTu:Optional[nn.Module]=None,
                FacTv:Optional[nn.Module]=None,
                FacTc:Optional[nn.Module]=None):
        qkv = self.qkv(x)

        new_q = FacTv(self.dp_q(self.q_Facts(FacTu(x))))

        new_v =FacTv(self.dp_v(self.v_Facts(FacTu(x))))

        qkv[:, :, :, :, : self.dim] += new_q
        qkv[:, :, :, :,  -self.dim:] += new_v 

        return qkv

class _Fact_tk(nn.Module):
    def __init__(self,
                 qkv:nn.Module,
                 q_Facts:nn.Module,
                 v_Facts:nn.Module,
                 ) -> None:

        super().__init__()

        self.qkv = qkv 

        self.q_Facts = q_Facts
        self.v_Facts = v_Facts

    
    def forward(self,x,FacTu:Optional[nn.Module]=None,
                FacTv:Optional[nn.Module]=None,
                FacTc:Optional[nn.Module]=None):
        qkv = self.qkv(x)

        down_q = FacTu(x)
        down_v = FacTu(x)

        down_q = torch.einsum("bdhwr,rtk->bdhwtk",[down_q,FacTc])
        down_v = torch.einsum("bdhwr,rtk->bdhwtk",[down_v,FacTc])
        
        new_q = FacTv(self.q_Facts(down_q).squeeze())

        new_v = FacTv(self.q_Facts(down_v).squeeze())

        qkv[:, :, :, :, : self.dim] += new_q
        qkv[:, :, :, :,  -self.dim:] += new_v 

        return qkv

if __name__ == "__main__":
    t = torch.randn(2,8,8,8,768)
    model = Project_layer(in_channels=768,hidden_channels=16,out_channels=768,act_fn="gelu",norm_layer=nn.LayerNorm,reset_params=True)
    # print(model)
    # print(model(t).shape)
    # grn = GRN_3D(768,"second")
    # print(grn(t).shape)