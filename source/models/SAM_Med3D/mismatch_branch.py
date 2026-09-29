from models.baseline.SENet import senet3d18,senet3d10
from models.baseline.repvit import repvit_m0_6
from models.baseline.TF_net import LightTF_Net
import torch.nn as nn
import torch
from typing import Optional, Union, Tuple
from einops import rearrange
from .model_util import Project_layer


# class h_sigmoid(nn.Module):
#     def __init__(self,inplace=True):
#         super().__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self,x):
#         return self.relu(x + 3) / 6
#
# class h_swish(nn.Module):
#     def __init__(self, inplace = True):
#         super().__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return x*output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

class Gate(nn.Module):
    """
    门控
    """
    def __init__(self, in_chans) -> None:
        super().__init__()
        self.linear = nn.Linear(in_chans, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = rearrange(x, "b d h w c -> b (d h w) c")
        x = torch.mean(x, dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1) # B 1 1 1 C
        return x

class Feature_extractor(nn.Module):
    """
    根据已有的senet网络，将其作为特征提取器，先从原始图像中对特征进行提取，然后再把差异特征进行相减
    """
    def __init__(self,in_channels=1,stride=2,stages=True):
        super().__init__()

        extractor = list(senet3d18(input_channels=in_channels,stride=stride).children())[:-2]

        self.stem = nn.Sequential(
            *extractor[:4]
        )

        self.layers = nn.ModuleList(
            extractor[4:]
        )

        self.stages = stages

    def forward(self,x):
        x_ls = []
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)
            x_ls.append(x)

        if not self.stages:
            return x
        else:
            return x_ls

class Feature_extractorv2(nn.Module):
    """
    根据已有的repvit网络，将其作为特征提取器，先从原始图像中对特征进行提取，然后再把差异特征进行相减
    """
    def __init__(self,in_channels=1,stages=True, need_layer=[1,4,14,16]):
        super().__init__()

        extractor = repvit_m0_6(in_chans=in_channels)

        self.stem = extractor.features[0]

        self.layers = extractor.features[1:]

        self.stages = stages
        self.need_layer = need_layer

    def forward(self,x):
        x_ls = []
        x = self.stem(x)

        for layer_index,layer in enumerate(self.layers):
            x = layer(x)
            if layer_index in self.need_layer:
                x_ls.append(x)

        if not self.stages:
            return x
        else:
            return x_ls

class Feature_extractorv3(nn.Module):
    """
    根据已有的repvit网络，将其作为特征提取器，先从原始图像中对特征进行提取，然后再把差异特征进行相减
    [0,2,4,6]
    [1,3,5,6]
    """
    def __init__(self,in_channels=1,stages=True, need_layer=[0,2,4,6]):
        super().__init__()

        extractor = LightTF_Net(in_chans=in_channels)

        self.stem = extractor.stem

        self.layers = extractor.features

        self.stages = stages
        self.need_layer = need_layer


    def forward(self,x):
        x_ls = []
        x = self.stem(x)

        for layer_index,layer in enumerate(self.layers):
            x = layer(x)
            if layer_index in self.need_layer:
                x_ls.append(x)

        if not self.stages:
            return x
        else:
            return x_ls



class Project_kv(nn.Module):
    """
    映射为k和v向量
    """
    def __init__(self, in_channels, out_channels, down_sample_stride=1, kv_bias=True, hidden_dim:Optional[int]=None, up_sample=False):
        super().__init__()
        # 对特征图进行下采样，同时将通道数变成目标通道数
        if up_sample:
            self.down_sample = nn.Sequential(
                # SpatialAttention(),
                # CBAMBlock(channel=in_channels),
                # nn.BatchNorm3d(in_channels),
                nn.ConvTranspose3d(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=down_sample_stride,
                    bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            # 特征图处理，对于浅层较大的feature，选择使用avgpool，然后使用卷积核大小为1的卷积来进行通道升维，要不然信息损失太多了
            self.down_sample = nn.Sequential(
                # SpatialAttention(),
                # CBAMBlock(channel=in_channels),
                # nn.BatchNorm3d(in_channels),
                # nn.Identity() if down_sample_stride == 1 else nn.AvgPool3d(kernel_size=down_sample_stride, stride=down_sample_stride),
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=down_sample_stride,
                    bias=False),
                nn.BatchNorm3d(out_channels)
            )
        if hidden_dim is None:
            self.project_kv = nn.Linear(out_channels, 2*out_channels,bias=kv_bias)
        else:
            self.project_kv = Project_layer(in_channels=out_channels,hidden_channels=hidden_dim,out_channels=2*out_channels,bias=False)

    def forward(self, x):
        x = self.down_sample(x)

        # # 参数位置编码
        # if pos_embed:
        #     x += pos_embed

        x = x.permute(0, 2, 3, 4, 1).contiguous()

        kv = self.project_kv(x)

        return kv



class Mismatch_extractor(nn.Module):
    def __init__(self,in_channels=1,stride=2,stages=True):
        super().__init__()

        self.feature_extractor0 = Feature_extractor(in_channels, stride, stages)
        self.feature_extractor1 = Feature_extractor(in_channels, stride, stages)

    def forward(self,x0,x1):

        f_x0 = self.feature_extractor0(x0)

        f_x1 = self.feature_extractor1(x1)

        mis_features = []

        for i in range(len(f_x0)):
            mis_features.append(f_x0[i] - f_x1[i])

        # 逆序 深层CNN与浅层VIT  深层VIT与浅层CNN
        # mis_features.reverse()

        return mis_features

class Mismatch_extractorV2(nn.Module):
    def __init__(self,in_channels=1,stride=2,stages=True):
        super().__init__()

        self.feature_extractor0 = Feature_extractorv3(in_channels, stages)
        self.feature_extractor1 = Feature_extractorv3(in_channels, stages)
        # self.feature_extractor0 = Feature_extractorv2(in_channels, stages)
        # self.feature_extractor1 = Feature_extractorv2(in_channels, stages)

    def forward(self,x0,x1):

        f_x0 = self.feature_extractor0(x0)

        f_x1 = self.feature_extractor1(x1)

        mis_features = []

        for i in range(len(f_x0)):
            mis_features.append(f_x0[i] - f_x1[i])

        # 逆序 深层CNN与浅层VIT  深层VIT与浅层CNN
        # mis_features.reverse()
        # 40 80 160 320  (32 16 8 4)
        # 128 128 256 192   16 8  4 4
        return mis_features

class Feature_extractor_MFE(nn.Module):
    def __init__(self,in_channels=1):
        super().__init__()
        extractor1 = LightTF_Net(in_chans=in_channels)
        extractor2 = LightTF_Net(in_chans=in_channels)

        self.img_level = extractor1
        self.feature_level = extractor2
    def forward(self,x1,x2):
        img_features = self.img_level(x1-x2)
        feature_level = self.feature_level(x1) - self.feature_level(x2)
        final_mis =  torch.cat([img_features,feature_level],dim=1)
        ls = [final_mis]*4
        return ls



if __name__ == "__main__":
    # model = CBAMBlock(channel=80)
    # ls = [2,5,8,11]
    # is_bool = []
    # # model= Feature_extractorv2()
    t1 = torch.randn(2, 1, 128, 128, 128)
    t2 = torch.randn(2, 1, 128, 128, 128)
    # ls = model(t)
    # # for i in range(12):
    # #     is_bool.append(True if i in ls and True else False)
    # # print(is_bool)
    model = Feature_extractor_MFE()
    f = model(t1,t2)  # 2 488
    params = sum([params.nelement() for params in model.parameters()])/1e6
    print(params)
    #
    # print(len())








