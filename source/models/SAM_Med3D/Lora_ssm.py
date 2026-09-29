#状态空间微调
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch.cuda.amp import autocast

# simple structure state space
# class sSSM(nn.Module):
#     def __init__(self):
#         pass
class Mamba_block(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class Mamba_adapter(nn.Module):
    def __init__(self,in_channels,hidden_channels,tri_orientate=False):
        super().__init__()
        self.norm0 = nn.LayerNorm(in_channels)
        self.down_sample = nn.Linear(in_channels,hidden_channels)
        self.act_fn = nn.GELU()
        self.conv_x = nn.Conv3d(hidden_channels,hidden_channels,groups=hidden_channels,kernel_size=3,padding=1)
        self.mamba = Mamba_block(dim=hidden_channels)
        self.up_sample = nn.Linear(hidden_channels,in_channels)

        self.tri_orientate = tri_orientate
        
    def forward(self,x):
        shortcut = x
        x = self.norm0(x)
        x = self.down_sample(x)
        x = self.act_fn(x)

        # shortcut2 = x

        x = x.permute(0,4,1,2,3).contiguous()

        x = self.conv_x(x)

        if self.tri_orientate:
            x_forward = x
            x_reverse = x.permute(0,1,4,3,2).contiguous()
            x_inter = x.permute(0,1,4,2,3).contiguous()

            x_forward = self.mamba(x_forward)
            x_reverse = self.mamba(x_reverse)
            x_inter = self.mamba(x_inter)

            x_reverse = x_reverse.permute(0,1,4,3,2).contiguous()
            x_inter = x_inter.permute(0,1,3,4,2).contiguous()

            x_out = x_forward + x_reverse + x_inter
        else:
            x_out = self.mamba(x)

        x = x_out.permute(0,2,3,4,1).contiguous()

        # x += shortcut2

        x = self.up_sample(x)

        return x + shortcut

class Basic_mamba_layer(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.convx = nn.Conv3d(in_channels,in_channels,kernel_size=3,groups=in_channels,padding=1)
        self.norm = nn.BatchNorm3d(in_channels)
        self.act_fn = nn.LeakyReLU()
        self.mamba = Mamba_block(dim=in_channels)

    def forward(self,x):
        shortcut = x
        x = self.convx(x)
        x = self.norm(x)
        x = self.act_fn(x)
        x = self.mamba(x)
        return x + shortcut

class PVM_mamba_adapter(nn.Module):
    def __init__(self,in_channels, split_num=4):
        super().__init__()

        assert in_channels % split_num == 0

        self.norm0 = nn.LayerNorm(in_channels)

        self.mamba_layers = nn.ModuleList(
            [Basic_mamba_layer(in_channels=in_channels//split_num) for i in range(split_num)]
        )

        self.norm1 = nn.BatchNorm3d(in_channels)

        self.fuse = nn.Conv3d(in_channels,in_channels,kernel_size=1,groups=in_channels)

        self.split_num = split_num

    def forward(self,x):
        residual = x
        B, H, W, D, C = x.shape
        splited_channel = C//self.split_num

        x = self.norm0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        mamba_outs = []

        for i, layer in enumerate(self.mamba_layers):
            mamba_outs.append(layer(x[:,i*splited_channel:(i+1)*splited_channel,...]))

        x_combined = torch.cat(mamba_outs,dim=1)

        x_combined = self.norm1(x_combined)

        x_combined = self.fuse(x_combined)

        x_combined = x_combined.permute(0, 2, 3, 4, 1).contiguous()

        return x_combined + residual

class Mamba_adapterv2(nn.Module):

    def __init__(self,in_channels,hidden_channels,split_num=4):
        super().__init__()

        self.norm = nn.LayerNorm(in_channels)
        self.down_sample = nn.Linear(in_channels, hidden_channels)
        self.pvm_mamba_adapter = PVM_mamba_adapter(in_channels=hidden_channels,split_num=split_num)
        self.up_sample = nn.Linear(hidden_channels, in_channels)

    def forward(self,x):
        x = self.norm(x)
        x = self.down_sample(x)
        x = self.pvm_mamba_adapter(x)
        x = self.up_sample(x)

        return x





if __name__ == "__main__":
    t = torch.randn(4,8,8,8,768).cuda()
    model = Mamba_adapter(in_channels=768,hidden_channels=128,tri_orientate=True).cuda()
    # model = PVM_mamba_adapter(in_channels=768,split_num=6).cuda()
    # t = torch.randn(4, 128 ,8,8,8).cuda()
    # model = Mamba_block(128).cuda()
    params = sum([params.nelement() for params in model.parameters()])/1e6
    print(params)
    print(model(t).shape)
