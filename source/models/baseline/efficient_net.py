from monai.networks.nets import EfficientNetBN

def cal_params(model):
    param_num = sum([params.nelement() for params in model.parameters()])
    return param_num/1e6

model = EfficientNetBN("efficientnet-b6",spatial_dims=3,in_channels=4,num_classes=2)
print(cal_params(model))