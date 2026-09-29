# TAFE Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR 

class After_fusion(nn.Module):
    def __init__(self,in_chans=48*4, hidden_chans=256, out_chans=48, activation=nn.LeakyReLU(inplace=True),norm_layer=nn.BatchNorm3d):
        super().__init__()

        self.down = nn.Conv3d(in_chans,hidden_chans,kernel_size=1)
        self.norm0 = norm_layer(hidden_chans)

        self.up = nn.Conv3d(hidden_chans, out_chans,kernel_size=1)
        self.norm1 = norm_layer(out_chans)

        self.act_fn = activation

        # 初始化参数的方式

    def forward(self,x):

        x = self.down(x)
        x = self.norm0(x)
        x = self.act_fn(x)

        x = self.up(x)
        x = self.norm1(x)
        x = self.act_fn(x)

        return x

class MultiModalPatchEmbed(nn.Module):
    def __init__(self,old_embed,after_fusion):
        super().__init__()
        self.patch_embed = old_embed
        self.after_fusion = after_fusion

    def forward(self,x):
        C = x.shape[1]
        ls = []
        # B D H W C
        for i in range(C):
            ls.append(self.patch_embed(x[:,i,:,:,:].unsqueeze(1)))
        x_ = torch.cat(ls,dim=1)
        x = self.after_fusion(x_)
        return  x

class TAFEModule(nn.Module):
    """
    Tumor-Aware Feature Encoding (TAFE) module.
    
    This module uses a SWIN-UNETR backbone to perform tumor segmentation
    and to extract multi-scale encoder features for classification. The segmentation
    branch produces segmentation logits (S ∈ R^(B×2×D×H×W)), while a global
    average pooling is applied to the encoder feature map to yield a compact 
    feature vector that is passed to a fully connected classification head.
    
    Parameters:
        backbone (nn.Module): Backbone segmentation network (default: SwinUNETR).
        img_size (tuple): Spatial dimensions of the input image.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for segmentation.
        feature_size (int): Base feature size for the backbone.
        depths (tuple): Depth (number of blocks) at each encoder stage.
        num_heads (tuple): Number of attention heads at each stage.
        classification_channels (int): Number of channels from the selected encoder features.
        num_classes (int): Number of classification output classes.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
        pretrained_path (str): Path to pretrained weights, if any.
        **kwargs: Additional arguments for the backbone.
    """
    def __init__(
        self,
        backbone: nn.Module = None,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=4,   # 0=BG,1=NCR/NET,2=ED,3=ET
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        classification_channels=None,
        num_classes=2,
        use_checkpoint=True,
        pretrained_path=None,
        **kwargs
    ):
        super().__init__()
        if backbone is None:
            self.backbone = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                depths=depths,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint,
                **kwargs
            )
        else:
            self.backbone = backbone

        # Determine classification channels
        if classification_channels is None:
            classification_channels = feature_size * 16

        # Classification head: map pooled features to final classes.
        # self.classification_head = nn.Sequential(
        #     nn.Linear(classification_channels, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(64, num_classes)
        # )

        # self.model_path = pretrained_path if pretrained_path is not None else ''
        check = torch.load("/home/cyx/Datasets/64-gpu-model_bestValRMSE.pt")
        model_check = check["state_dict"]
        for k in list(model_check.keys()):
            if "swinViT" in k:
                model_check[k[len("module."):]] = model_check[k]
            del model_check[k]

        msg = self.backbone.load_state_dict(model_check,strict=False)
        # print(msg)
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        self.backbone.swinViT.patch_embed = MultiModalPatchEmbed(self.backbone.swinViT.patch_embed,After_fusion())
        


    def forward(self, x_in):
        """
        Forward pass.
        
        Args:
            x_in (tensor): Input tensor of shape [B, in_channels, D, H, W].
            
        Returns:
            seg_logits: Segmentation logits of shape [B, 2, D, H, W].
            cls_logits: Classification logits of shape [B, num_classes].
        """
        # Obtain hierarchical features from the backbone's Swin transformer encoder.
        hidden_states_out = self.backbone.swinViT(x_in, normalize=True)

        # --- Segmentation branch ---
        # enc0 = self.backbone.encoder1(x_in)
        # enc1 = self.backbone.encoder2(hidden_states_out[0])
        # enc2 = self.backbone.encoder3(hidden_states_out[1])
        # enc3 = self.backbone.encoder4(hidden_states_out[2])
        # dec4 = self.backbone.encoder10(hidden_states_out[4])
        # dec3 = self.backbone.decoder5(dec4, hidden_states_out[3])
        # dec2 = self.backbone.decoder4(dec3, enc3)
        # dec1 = self.backbone.decoder3(dec2, enc2)
        # dec0 = self.backbone.decoder2(dec1, enc1)
        # out = self.backbone.decoder1(dec0, enc0)
        # seg_logits = self.backbone.out(out)  # [B, 4, D, H, W]
        # seg_logits,

        # --- Classification branch ---
        x_deep = hidden_states_out[4]  # [B, classification_channels, D', H', W']
        x_pooled = F.adaptive_avg_pool3d(x_deep, (1, 1, 1)).view(x_in.size(0), -1)
        return x_pooled
        # cls_logits = self.classification_head(x_pooled)

        # return  cls_logits
