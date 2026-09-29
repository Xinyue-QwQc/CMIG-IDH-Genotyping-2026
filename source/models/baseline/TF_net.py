import torch
import torch.nn as nn
import torch.nn.functional as F
import monai.networks.nets as nets




# Come from repvit
class Conv3d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv3d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm3d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv3d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class SE_Layer(nn.Module):
    def __init__(self,channels,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels,channels // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )

    def forward(self,x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1,1)
        return x * y


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed, flag = False) -> None:
        super().__init__()

        self.conv = Conv3d_BN(ed, ed, 5 if flag else 3, 1, 2 if flag else 1, groups=ed)
        self.conv1 = nn.Conv3d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = nn.BatchNorm3d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

class RepVGGDW_(torch.nn.Module):
    def __init__(self, ed, flag = False) -> None:
        super().__init__()

        self.conv = Conv3d_BN(ed, 2*ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv3d(ed, ed, 1, 1, 0, groups=ed)
        self.act_fn = nn.ReLU6()
        self.dim = ed
        self.bn = nn.BatchNorm3d(ed)


    def forward(self, x):
        residual = x
        xz = self.conv(x)
        x,z = xz.chunk(2,dim=1)
        x = x * self.act_fn(z)
        # x = self.conv1(x)
        return self.bn(x + self.conv1(residual) + residual)
# class Residual(torch.nn.Module):
#     def __init__(self, m, drop=0.):
#         super().__init__()
#         self.m = m
#         self.drop = drop
#
#     def forward(self, x):
#         if self.training and self.drop > 0:
#             return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, 1,
#                                               device=x.device).ge_(self.drop).div(1 - self.drop).detach()
#         else:
#             return x + self.m(x)

class TF_layer(nn.Module):
    def __init__(self,inp, bn_size, flag, use_se=True, growth_rate=32,drop_rate=0):
        super().__init__()
        oup = bn_size * growth_rate
        self.token_mixer = nn.Sequential(
            RepVGGDW(inp, flag),
            SE_Layer(inp,4) if use_se else nn.Identity()
        )
        self.channel_mixer = nn.Sequential(
                Conv3d_BN(inp,oup,1,1,0),
                nn.GELU(),
                Conv3d_BN(oup,growth_rate,1,1,0,bn_weight_init=0)
            )

        self.drop_rate = drop_rate

    def forward(self,x):
        new_feature = self.channel_mixer(self.token_mixer(x))
        if self.drop_rate > 0:
            new_feature = F.dropout(new_feature,
                                     p=self.drop_rate,
                                     training=self.training)

        return torch.cat([x,new_feature],dim=1)

class TF_Block(nn.Sequential):
    def __init__(self,num_layers,num_input_features,bn_size,growth_rate,drop_rate,flag):
        super().__init__()
        for i in range(num_layers):
            layer = TF_layer(num_input_features + i*growth_rate,
                             growth_rate=growth_rate, bn_size=bn_size,
                             drop_rate=drop_rate, flag=flag)
            self.add_module("tf_layer{}".format(i+1), layer)

class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('gelu', nn.GELU())
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))




class LightTF_Net(nn.Module):
    #(2,2,6,2)  (3,3,9,3)
    # (3,4,6,3)  resnet34的block config
    def __init__(self, in_chans,  num_init_feature = 64, block_config = (3,4,6,3),
                 growth_rate = 32, bn_size = 4, flag=True, drop_rate=0,
                 make_cls=False, num_classes=2):
        
        super().__init__()

        # 一开始用大一点的核进行卷积特征提取
        self.stem = nn.Sequential(Conv3d_BN(in_chans, num_init_feature // 2, 5, 2, 2), nn.GELU(),
                                          Conv3d_BN(num_init_feature // 2, num_init_feature, 5, 2, 2))

        self.features = nn.Sequential()

        self.make_cls = make_cls



        num_feature = num_init_feature

        for i, num_layers in enumerate(block_config):

            if i > 1:
                flag = False

            block = TF_Block(num_layers=num_layers,
                             num_input_features=num_feature,
                             bn_size=bn_size,
                             growth_rate=growth_rate,
                             drop_rate=drop_rate,
                             flag=flag
                             )
            self.features.add_module('TF_block{}'.format(i+1),block)
            num_feature = num_feature + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_feature,
                    num_output_features=num_feature // 2
                )
                self.features.add_module("transition{}".format(i+1),trans)
                num_feature = num_feature // 2

        if make_cls:
            self.classifier = nn.Linear(num_feature,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # m.weight = nn.init.kaiming_normal()
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self,x):
        x = self.stem(x)
        x = self.features(x)

        if self.make_cls:
            x = F.adaptive_avg_pool3d(x,1).flatten(1)
            x = self.classifier(x)
            return x
        return x

if __name__ == "__main__":

    model = LightTF_Net(in_chans=1,make_cls=False)
    # print(model)
    # t = torch.randn(2,4,128,128,128)
    # print(model(t).shape)
    # need_layer = [0,2,4,6]
    # print(len(model.features))
    # for i,layer in enumerate(model.features):
    #     if i in need_layer:
    #         print(i,layer)
    # print(model(t).shape)
    param = sum([p.numel() for p in model.parameters()])
    print(param / 1e6)