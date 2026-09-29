import torch 
import torch.nn as nn 
from einops import rearrange
import math

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
    
class Mismatch_extractv1(nn.Module):
    def __init__(self,in_chans=1,hidden_chans=96,embed_dim=384,activation=nn.GELU) -> None:
        super().__init__()
        self.simple_extracter = nn.Sequential(
            nn.Conv3d(in_chans, hidden_chans // 4, kernel_size=2, stride=2),
            LayerNorm3d(hidden_chans // 4),
            activation(),
            nn.Conv3d(hidden_chans // 4, hidden_chans//4, kernel_size=2, stride=2),
            LayerNorm3d(hidden_chans // 4),
            activation(),
            nn.Conv3d(hidden_chans // 4, hidden_chans, kernel_size=2, stride=2),
            LayerNorm3d(hidden_chans),
            activation(),
            nn.Conv3d(hidden_chans, embed_dim//2, kernel_size=1),
        )
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)
    def forward(self,x1,x2):
        mismatch = x1 - x2
        mismatch = self.simple_extracter(mismatch)
        avg_mis = self.avg_pool(mismatch)
        max_mis = self.max_pool(mismatch)
        mis = torch.cat([avg_mis,max_mis],dim=1)
        return mis 

class Gate(nn.Module):
    def __init__(self,in_chans) -> None:
        super().__init__()
        self.linear = nn.Linear(in_chans,1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self,x):
        x = x.permute(0,2,3,4,1).contiguous()
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = rearrange(x,"b d h w c -> b (d h w) c")
        x = torch.mean(x,dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return x 

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y



class Basic_layer(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,norm_layer=nn.BatchNorm3d,act_fn=nn.GELU,drop=0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,hidden_channels,kernel_size=1)
        self.bn1 = norm_layer(hidden_channels)

        self.conv2 = nn.Conv3d(hidden_channels,hidden_channels,kernel_size=3,padding=1,groups=hidden_channels)
        self.bn2 = norm_layer(hidden_channels)

        self.conv3 = nn.Conv3d(hidden_channels,out_channels,kernel_size=1)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels:
            self.down = nn.Conv3d(in_channels,out_channels,kernel_size=1)
        else:
            self.down = None


        self.act_fn = act_fn()
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        residual = x 
        x = self.bn1(self.conv1(x))
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.bn2(self.conv2(x))
        x = self.act_fn(x)

        x = self.bn3(self.conv3(x))
        x = self.drop(x)

        if self.down is not None:
            x += self.down(residual)
        else:
            x += residual
        return x 

class Basic_layerv2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



class conv_pooling(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels,out_channels,kernel_size=2,stride=2)
    
    def forward(self,x):
        return self.conv(x)

class Feature_extracterv2(nn.Module):
    def __init__(self,dims) -> None:
        super().__init__()
        # 首先对两个模态特征进行提取，然后再对提取后的特征进行相减
        # 之后再加上一个门控机制控制特征的情况
        self.layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.depths = len(dims)

        self.layers.append(Basic_layer(1,dims[0],dims[0]))

        for i in range(self.depths-1):
            self.pool_layers.append(conv_pooling(dims[i],dims[i+1]))
            self.layers.append(Basic_layer(dims[i+1],dims[i+1],dims[i+1]))
        
        self.pool_layers.append(conv_pooling(dims[-1],dims[-1]*2))
               
        
    def forward(self,x):
        x = self.layers[0](x)
        for i in range(self.depths-1):
            x = self.pool_layers[i](x)
            x = self.layers[i+1](x)
        # B 256 8 8 8 
        x = self.pool_layers[-1](x)
          
        return x 
    
# 有一个idea是直接在频率域上提取特征，然后再利用全局和局部的方式进行提取，再分离
class Mismatch_extracterv2(nn.Module):
    def __init__(self,dims,final_channles,h=8,w=8,d=8) -> None:
        super().__init__()

        self.extracter1 = Feature_extracterv2(dims)
        self.extracter2 = Feature_extracterv2(dims)
        self.se_layer = SELayer(dims[-1]*2)

        self.new_d = d//2 + 1
        self.complex_weight = nn.Parameter(torch.randn(dims[-1]*2,h,w,self.new_d,2,dtype=torch.float32)*0.02)

        self.gate = Gate(dims[-1]*2)
        self.up = nn.Conv3d(dims[-1]*2,final_channles,kernel_size=1)

        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out")
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight,a=math.sqrt(5))
       
    
    def forward(self,x1,x2):
    
        x1 = self.extracter1(x1)
        x2 = self.extracter2(x2)

        # single path 
        x_mis = x1 - x2 
        # x_mis = self.extracter1(x_mis)
        x_gate = self.gate(x_mis)
        x_mis = self.se_layer(x_mis)

        B,C,D,H,W = x_mis.shape

        x_mis_signs = torch.fft.rfftn(x_mis,dim=(2,3,4),norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_mis_signs = x_mis_signs * weight
        x_mis = torch.fft.irfftn(x_mis_signs,s=(D,H,W),dim=(2,3,4),norm="ortho")
        x_mis = self.up(x_mis)

        return x_mis*x_gate





# class Feature_extracterv3(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.f


        


if __name__ == "__main__":
    x1 = torch.randn(2,1,128,128,128)
    x2 = torch.randn(2,1,128,128,128)
    
    dims = [16,32,64,128]
    model = Mismatch_extracterv2(dims,384)
    # x = model(x1)
    # print(x.shape)
    params = sum([params.numel() for params in model.parameters()])
    print(params/1e6)
    mis = model(x1,x2)
    print(mis.shape)