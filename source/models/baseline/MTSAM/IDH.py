import torch.nn as nn
class IDH(nn.Module):
    def __init__(self):
        super(IDH, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(384, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        self.drop=nn.Dropout(0.1)
        self.relu=nn.LeakyReLU()

    def forward(self, x):
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x=self.drop(x)
        # 全连接层
        x = self.fc1(x)
        x=self.relu(x)
        x=self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x=self.drop(x)
        x = self.fc3(x)
        return x