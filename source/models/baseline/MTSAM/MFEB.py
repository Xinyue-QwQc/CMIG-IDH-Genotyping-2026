
import torch.nn as nn
import torch
import torch.nn.functional as F


class MTC3(nn.Module):
    def __init__(self, intc, outc, dropout_prob=0.1):
        super(MTC3, self).__init__()

        # 第一层卷积：3x3x3，padding=1，dilation=1
        self.conv1 = nn.Conv3d(in_channels=intc, out_channels=intc, kernel_size=3, padding=3, dilation=3)

        # LeakyReLU, BatchNorm 和 Dropout
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.bn1 = nn.BatchNorm3d(intc)
        self.dropout1 = nn.Dropout3d(p=dropout_prob)

        # 第二层卷积：3x3x3，padding=1，dilation=1
        self.conv21 = nn.Conv3d(in_channels=intc, out_channels=outc, kernel_size=3, padding=1, dilation=1)
        self.conv22 = nn.Conv3d(in_channels=intc, out_channels=outc, kernel_size=3, padding=1, dilation=1)
        # 两个 1x1x1 的卷积（FFN）
        self.conv3 = nn.Conv3d(in_channels=outc, out_channels=outc, kernel_size=1)
        self.conv4 = nn.Conv3d(in_channels=outc, out_channels=outc, kernel_size=1)

    def forward(self, x):
        short=x
        # 1. 扩张卷积：提取较大范围的特征
        x = self.conv1(x)
        x = self.act1(self.bn1(x))  # 激活 + 归一化
        x = self.dropout1(x)  # Dropout

        # 2. 第二个卷积：减少通道数到1
        x = self.conv21(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.conv22(x)

        # 3. FFN 部分：两个 1x1x1 卷积
        x = self.conv3(x)
        x=self.dropout1(x)
        x=self.act1(x)
        x = self.conv4(x)
        x=short+x

        return x

class MTC5(nn.Module):
    def __init__(self, intc, outc, dropout_prob=0.1):
        super(MTC5, self).__init__()
        # 重新计算填充值以保持分辨率不变
        padding = (3 * (5 - 1)) // 2
        self.conv1 = nn.Conv3d(in_channels=intc, out_channels=intc, kernel_size=5, padding=padding, dilation=3)

        # LeakyReLU, BatchNorm 和 Dropout
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.bn1 = nn.BatchNorm3d(intc)
        self.dropout1 = nn.Dropout3d(p=dropout_prob)

        # 重新计算填充值以保持分辨率不变
        padding = (5 - 1) // 2
        self.conv21 = nn.Conv3d(in_channels=intc, out_channels=outc, kernel_size=5, padding=padding)
        self.conv22 = nn.Conv3d(in_channels=intc, out_channels=outc, kernel_size=5, padding=padding)

        # 两个 1x1x1 的卷积（FFN）
        self.conv3 = nn.Conv3d(in_channels=outc, out_channels=outc, kernel_size=1)
        self.conv4 = nn.Conv3d(in_channels=outc, out_channels=outc, kernel_size=1)

    def forward(self, x):
        # 1. 扩张卷积：提取较大范围的特征
        short=x
        x = self.conv1(x)
        x = self.act1(self.bn1(x))  # 激活 + 归一化
        x = self.dropout1(x)  # Dropout

        # 2. 第二个卷积：减少通道数到1
        x = self.conv21(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.conv22(x)

        # 3. FFN 部分：两个 1x1x1 卷积
        x = self.conv3(x)
        x=self.dropout1(x)
        x=self.act1(x)
        x = self.conv4(x)
        x=short+x

        return x


class MTC7(nn.Module):
    def __init__(self, intc, outc, dropout_prob=0.1):
        super(MTC7, self).__init__()

        # 第一个 7x7x7 扩张卷积层，用于捕获更大的上下文信息，dilation=3
        self.conv1 = nn.Conv3d(in_channels=intc, out_channels=outc, kernel_size=7, padding=9, dilation=3)

        # LeakyReLU, BatchNorm 和 Dropout
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.bn1 = nn.BatchNorm3d(outc)
        self.dropout1 = nn.Dropout3d(p=dropout_prob)


        self.conv21 = nn.Conv3d(in_channels=outc, out_channels=outc, kernel_size=7, padding=3)
        self.conv22 = nn.Conv3d(in_channels=outc, out_channels=outc, kernel_size=7, padding=3)

        self.conv3 = nn.Conv3d(in_channels=outc, out_channels=outc, kernel_size=1)
        self.conv4 = nn.Conv3d(in_channels=outc, out_channels=outc, kernel_size=1)

    def forward(self, x):
        # 1. 扩张卷积：提取较大范围的特征

        x = self.conv1(x)
        x = self.act1(self.bn1(x))  # 激活 + 归一化
        x = self.dropout1(x)  # Dropout

        # 2. 第二个卷积：减少通道数到1
        x=self.conv21(x)
        x=self.act1(x)
        x=self.dropout1(x)
        x = self.conv22(x)

        # 3. FFN 部分：两个 1x1x1 卷积
        x = self.conv3(x)
        x=self.dropout1(x)
        x=self.act1(x)
        x = self.conv4(x)

        return x


class mfeb(nn.Module):
    def __init__(self):
        super(mfeb, self).__init__()
        self.mtc7 = MTC7(intc=2,outc=16)
        self.mtc5 = MTC5(intc=16,outc=16)
        self.mtc3 = MTC3(intc=16,outc=16)
        self.lp = nn.ConvTranspose3d(
            in_channels=16,  # 输入通道数，你可以根据实际情况修改
            out_channels=1,  # 输出通道数，你可以根据实际情况修改
            kernel_size=3,
            stride=2,  # 步长设置为2可以使分辨率变为原来的两倍
            padding=1,  # 填充值设置为1，配合步长和核大小保证输出尺寸合适
            output_padding=1  # 额外的输出填充，确保输出尺寸正确
        )
        self.sig = nn.Sigmoid()
        self.bat = nn.BatchNorm3d(1)
        self.relu=nn.LeakyReLU()

    def forward(self, x, y):
        s_x = x
        s_y = y
        x_half_size = tuple([s // 2 for s in x.shape[2:]])  # 计算x的空间尺寸的一半
        y_half_size = tuple([s // 2 for s in y.shape[2:]])  # 计算y的空间尺寸的一半

        # 自适应池化到输入的尺寸的一半
        x_max = F.adaptive_max_pool3d(x, output_size=x_half_size)  # 池化后尺寸变为输入尺寸的一半
        x_avg = F.adaptive_avg_pool3d(x, output_size=x_half_size)  # 池化后尺寸变为输入尺寸的一半
        y_max = F.adaptive_max_pool3d(y, output_size=y_half_size)  # 池化后尺寸变为输入尺寸的一半
        y_avg = F.adaptive_avg_pool3d(y, output_size=y_half_size)  # 池化后尺寸变为输入尺寸的一半

        # 在通道维度上拼接最大池化和平均池化的结果
        x = torch.cat([x_max, x_avg], dim=1)
        y = torch.cat([y_max, y_avg], dim=1)
        x = self.mtc7(x)
        y = self.mtc7(y)
        x = self.mtc5(x)
        y = self.mtc5(y)
        x = self.mtc3(x)
        y = self.mtc3(y)
        x=self.lp(x)
        y=self.lp(y)
        x = self.sig(x)
        y = self.sig(y)
        x = s_x * x
        y = s_y * y
        x = x - y
        x = self.bat(x)
        return x