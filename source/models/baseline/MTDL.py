import torch
import torch.nn as nn

class CNN_Block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel=3,is_pool=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel,padding=kernel//2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool3d(kernel_size=2) if is_pool else nn.Identity()

    def forward(self,x):
        x = self.relu(self.bn(self.conv(x)))
        return self.pool(x)

class MTDL(nn.Module):

    def __init__(self,in_chans,num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            CNN_Block(in_channels=in_chans,out_channels=128),
            CNN_Block(in_channels=128,out_channels=128,is_pool=True)
        )
        self.layer2 = nn.Sequential(
            CNN_Block(in_channels=128,out_channels=256),
            CNN_Block(in_channels=256, out_channels=256, is_pool=True)
        )
        self.layer3 = nn.Sequential(
            CNN_Block(in_channels=256, out_channels=256),
            CNN_Block(in_channels=256, out_channels=256),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.layer4 = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )
        self.classifier = nn.Linear(34,num_classes)

    def forward(self, x, age, gender):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0),-1)
        x = self.layer4(x)
        x = torch.cat([x,age,gender],dim=-1)
        y = self.classifier(x)
        return y

if __name__ == "__main__":
    model = MTDL(in_chans=4,num_classes=2)
    t = torch.randn(2,4,128,128,128)
    age = torch.randn(2,1)
    gender = torch.randn(2,1)
    y = model(t,age,gender)
    print(y.shape)
    params = sum([p.numel() for p in model.parameters()])
    print(params/1e6)

