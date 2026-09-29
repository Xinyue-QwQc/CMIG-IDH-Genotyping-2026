import torch.nn as nn
# import torch.nn.functional as F
import torch


class ACB(nn.Module):
    def __init__(self,in_chans,kernel_size,stride=1,is_pad=True):
        super().__init__()

        self.acb_conv = nn.Sequential(
            nn.Conv3d(in_chans, in_chans, kernel_size=(kernel_size, 1, 1), padding=(kernel_size//2, 0, 0) if is_pad else 0, stride=(stride, 1, 1)),
            nn.Conv3d(in_chans, in_chans, kernel_size=(1, kernel_size, 1), padding=(0, kernel_size // 2, 0) if is_pad else 0, stride=(1, stride, 1)),
            nn.Conv3d(in_chans, in_chans, kernel_size=(1, 1, kernel_size), padding=(0, 0, kernel_size // 2) if is_pad else 0, stride=(1, 1, stride)),
        )

    def forward(self, x):
        return self.acb_conv(x)

class BN_PR(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.bn_pr = nn.Sequential(
            nn.BatchNorm3d(dim),
            nn.PReLU()
        )
    def forward(self,x):
        return self.bn_pr(x)


class MBConv(nn.Module):
    def __init__(self,in_chans,out_chans):
        super().__init__()

        self.conv0 = nn.Conv3d(in_chans,out_chans,kernel_size=3,padding=1)
        self.bn_pr0 = BN_PR(out_chans)

        self.branch1 = nn.Sequential(
            nn.Conv3d(out_chans, out_chans//2,kernel_size=1),
            ACB(out_chans//2 ,kernel_size=3,stride=2),
            BN_PR(out_chans//2)
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(out_chans, int(out_chans*0.375), kernel_size=1),
            ACB(int(out_chans*0.375),  kernel_size=3, stride=1),
            ACB(int(out_chans*0.375), kernel_size=3, stride=2)
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(out_chans,int(out_chans*0.125), kernel_size=1),
            nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        )

    def forward(self,x):
        x = self.bn_pr0(self.conv0(x))

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat([x1,x2,x3],dim=1)

        return out

class C_P(nn.Module):
    def __init__(self,in_chans,out_chans=128):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_chans,out_chans, kernel_size=1),
            ACB(out_chans,kernel_size=3),
            ACB(out_chans, kernel_size=3, stride=2)
        )
        self.pool = nn.MaxPool3d(kernel_size=3,stride=2,padding=1)

        self.conv1 = nn.Sequential(
            ACB(out_chans, kernel_size=3),
            BN_PR(out_chans),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )


    def forward(self,x):
        x = self.conv0(x)
        x_ = self.conv1(x)
        x0 = self.pool(x)

        out = torch.cat([x_,x0],dim=1)
        return out

class MB_pool(nn.Module):
    def __init__(self,in_chans,out_chans=256):
        super().__init__()

        branch1_dim = int(out_chans*0.375)
        branch2_dim = int(out_chans*0.5)
        branch3_dim = int(out_chans*0.125)

        self.branch0 = nn.Sequential(
            nn.Conv3d(in_chans,branch1_dim,kernel_size=1),
            BN_PR(branch1_dim),
            nn.AdaptiveAvgPool3d(1)
        )

        self.branch1 = nn.Sequential(
            nn.Conv3d(in_chans,branch2_dim, kernel_size=1),
            ACB(branch2_dim,  kernel_size=3),
            ACB(branch2_dim,  kernel_size=3),
            ACB(branch2_dim, kernel_size=3),
            BN_PR(branch2_dim),
            nn.AdaptiveAvgPool3d(1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_chans,branch3_dim,kernel_size=1),
            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self, x):
        branch1 = self.branch0(x)
        branch2 = self.branch1(x)
        branch3 = self.branch2(x)

        x = torch.cat([branch1,branch2,branch3],dim=1)

        return x

class DAMM(nn.Module):
    def __init__(self,in_chans,out_chans=32,img_size=64):
        super().__init__()

        branch1_dim = int(out_chans*0.375)
        branch2_dim = int(out_chans*0.5)
        branch3_dim = int(out_chans*0.125)

        self.branch1 = nn.Sequential(
            nn.Conv3d(in_chans, int(out_chans*0.375), kernel_size=1),
            # ACB(int(out_chans*0.375), kernel_size=3, stride=1),
            ACB(int(out_chans*0.375), kernel_size=img_size,is_pad=False),
            BN_PR(branch1_dim)
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_chans, out_chans//2,kernel_size=1),
            ACB(out_chans//2, kernel_size=3, stride=1),
            ACB(out_chans//2, kernel_size=img_size,stride=1,is_pad=False),
            BN_PR(out_chans//2)
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(in_chans, int(out_chans*0.125), kernel_size=1),
            nn.AdaptiveAvgPool3d(1)
        )

        self.after_conv = nn.Sequential(
            nn.Conv3d(kernel_size=1,in_channels=out_chans,out_channels=out_chans//4),
            nn.PReLU(),
            nn.Conv3d(in_channels=out_chans//4,out_channels=in_chans,kernel_size=1)
        )

        self.conv_pool = nn.Sequential(
            ACB(in_chans=2, kernel_size=3),
            ACB(in_chans=2, kernel_size=3),
            ACB(in_chans=2, kernel_size=3),
            ACB(in_chans=2, kernel_size=3),
            nn.Conv3d(in_channels=2,out_channels=1,kernel_size=3,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        raw_x = x

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        b_out = torch.cat([b1,b2,b3],dim=1)
        ch_attn = self.after_conv(b_out)

        x = raw_x * ch_attn

        x_avg = torch.mean(x,dim=1,keepdim=True)
        x_max,_ = torch.max(x,dim=1,keepdim=True)

        x_sp = torch.cat([x_avg,x_max],dim=1)
        x_sp = self.conv_pool(x_sp)
        out = x*x_sp

        return out

class Main_backbone(nn.Module):
    def __init__(self,in_chans,out_chans):
        super().__init__()
        self.layer0 = nn.Sequential(
            MBConv(in_chans,out_chans),
            MBConv(out_chans,out_chans)
        )

        self.layer1 = nn.ModuleList(
            [
                C_P(out_chans,out_chans//2)
                for i in range(3)
            ]
        )
        
        self.mb_pool = MB_pool(out_chans,out_chans)
    
    def forward(self,x):
        x = self.layer0(x)
        ls = []
        
        for cp in self.layer1:
            x = cp(x)
            ls.append(x)
            
        y = self.mb_pool(x)
        ls.append(y)
        
        return ls 

class Fusion_net(nn.Module):
    def __init__(self,in_chans):
        super().__init__()
        self.conv = nn.Conv3d(in_chans, in_chans//4, kernel_size=1)
        self.pool = MB_pool(in_chans//4,in_chans//4)

    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class Aggn(nn.Module):
    def __init__(self,in_chans=4,hidden_chans=256,num_classes=2):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_chans,in_chans,kernel_size=5,padding=2,stride=2)

        self.layer0 = DAMM(in_chans)

        self.main_layer = nn.ModuleList(
            [
                Main_backbone(in_chans//4,hidden_chans)
                for i in range(4)
            ]
        )

        self.fusion_layer = nn.ModuleList(
            [
                Fusion_net(hidden_chans*4)
                for i in range(3)
            ]
        )

        self.classifier = nn.Linear(hidden_chans*7,num_classes)

    def forward(self,x):
        x = self.patch_embed(x)
        x = self.layer0(x)

        ls = torch.chunk(x, chunks=4, dim=1)

        results = []

        for i,sub_layer in enumerate(self.main_layer):
            results.append(sub_layer(ls[i]))

        main_x = torch.cat([results[i][-1] for i in range(len(results))], dim=1)
        main_x = main_x.view(main_x.size(0),-1)

        fusion_ls = []

        for i in range(3):
            fusion_ls.append(torch.cat([results[j][i] for j in range(4)],dim=1))

        for i,f_layer in enumerate(self.fusion_layer):
            fusion_ls[i] = f_layer(fusion_ls[i])

        fusion_x = torch.cat(fusion_ls,dim=1)
        fusion_x = fusion_x.view(fusion_x.size(0),-1)

        final_info = torch.cat([main_x,fusion_x],dim=1)

        y = self.classifier(final_info)

        return y

if __name__ == "__main__":
    model = Aggn()
    t = torch.randn((4,4,128,128,128))
    print(model(t).shape)
    # 20.31M
    params = sum([param.nelement() for param in model.parameters()])/1e6
    print(params)




