from models.baseline.MTSAM.ImageEncoderViT3D import ImageEncoderViT3D
from models.baseline.MTSAM.IDH import IDH
from models.baseline.MTSAM.MFEB import mfeb
import torch
import torch.nn as nn

class MFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc = nn.Linear(4,101)
    
    def forward(self, x):
        x = self.conv(x) 
        x = torch.mean(x, dim=(2, 3, 4))
        return self.fc(x)



class MTSAM(nn.Module):

    def __init__(self, check_path="/home/cyx/Datasets/sam_med3d_brain.pth",
                 r=4, after_fusion_channels=128,
                 in_channels=4, conv_adapter=False, Adapter_former=False,
                 parallel=True, mismatch=False, ConvMoe=False, text_embed=False) -> None:
        super().__init__()

        self.mismatch = mismatch
        # 导入预训练模型  仅仅使用encoder的
        check = torch.load(check_path, map_location="cpu")
        model_params = check["model_state_dict"]

        # 方便导入 去除前缀 同时去除prompt encoder和decoder的参数
        for k in list(model_params.keys()):

            if k.startswith("image_encoder"):
                model_params[k[len("image_encoder."):]] = model_params[k]

            del model_params[k]
        

        original_weight =  model_params['patch_embed.proj.weight']

        # 复制并平均权重
        with torch.no_grad():
            new_weight = original_weight.repeat(1, 5, 1, 1, 1)
        # 更新状态字典
        model_params['patch_embed.proj.weight'] = new_weight

        image_encoder = ImageEncoderViT3D(img_size=128, patch_size=16, out_chans=384, global_attn_indexes=[2, 5, 8, 11],
                                          use_rel_pos=True, window_size=14)

        msg = image_encoder.load_state_dict(model_params, strict=False)


        # 冻结所有参数
        for param in image_encoder.parameters():
            param.requires_grad = False

        for block in image_encoder.blocks:
            if hasattr(block, 'attn'):
                for param in block.attn.adapter.parameters():
                    param.requires_grad = True

        for name, param in image_encoder.named_parameters():
            if 'patch_embed' in name:
                param.requires_grad = True
        
        self.image_encoder = image_encoder
        self.t2f = mfeb()
        self.cls_net = IDH()
        self.manuscripture_net = MFE()



    def forward(self, x):
        # 101维的手工特征
        flair_image = x[:, 2, :, :, :].unsqueeze(1)
        t2_image = x[:, 3, :, :, :].unsqueeze(1)
        z = self.manuscripture_net(x)
        t2f = self.t2f(flair_image, t2_image)

        en_features = self.image_encoder(x, t2f, z)

        y = self.cls_net(en_features)
        return y 

if __name__ == "__main__":
    model = MTSAM()
    total = 0
    all_params = 0
    for name, values in model.named_parameters():
        if values.requires_grad:
            print(name)
            total += values.nelement()
        all_params += values.nelement()
    print(total / 1e6)
    print(total * 100 / all_params)
    print("ALL", all_params / 1e6)
    t = torch.randn(2, 4, 128, 128, 128)
    print(model(t).shape)