import pickle
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from monai.networks.nets.senet import SEResNext50
from monai.networks.layers.factories import Norm
from monai.utils import ensure_tuple_rep
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from einops import rearrange

class MLP(torch.nn.Module):
    def __init__(self, in_features):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.fc1 = torch.nn.Linear(in_features, 512)
        self.norm = Norm[Norm.BATCH, 1](num_features=512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.relu = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, self.in_features)  # Flatten the input
        x = self.fc2(self.norm(self.fc1(x)))
        x = self.fc3(self.relu(x))
        return x
  
class SEResNext50Single(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imaging_backbone = SEResNext50(in_channels=1, num_classes=512, spatial_dims=3)
        self.imaging_norm= Norm[Norm.BATCH, 1](num_features=512)
        self.relu = torch.nn.ReLU()
        self.feedforward = torch.nn.Linear(512, 512)
        self.classifier = torch.nn.Linear(512, 1)
    
    def forward(self, imgs):
        return self.classifier(self.relu(self.feedforward(self.imaging_norm(self.imaging_backbone(imgs)))))


class SEResNext50MML(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imaging_backbone = SEResNext50(in_channels=1, num_classes=512, spatial_dims=3)
        self.clinical_backbone = torch.nn.Linear(24, 512)
        self.imaging_norm= Norm[Norm.BATCH, 1](num_features=512)
        self.clinical_norm= Norm[Norm.BATCH, 1](num_features=512)
        self.relu = torch.nn.ReLU()
        self.feedforward = torch.nn.Linear(1024, 512)
        self.classifier = torch.nn.Linear(512, 1)
    
    def forward(self, imgs, clinical):
        return self.classifier(self.relu(self.feedforward(torch.cat([self.imaging_norm(self.imaging_backbone(imgs)), self.clinical_norm(self.clinical_backbone(clinical))], dim=1))))

class SEResNext50MMLALL(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ncct_imaging_backbone = SEResNext50(in_channels=2, num_classes=512, spatial_dims=3)
        self.ncct_norm= Norm[Norm.BATCH, 1](num_features=512)
        self.cta_imaging_backbone = SEResNext50(in_channels=2, num_classes=512, spatial_dims=3)
        self.cta_norm= Norm[Norm.BATCH, 1](num_features=512)
        self.clinical_backbone = torch.nn.Linear(2, 512)
        self.clinical_norm= Norm[Norm.BATCH, 1](num_features=512)
        self.relu = torch.nn.ReLU()
        self.feedforward = torch.nn.Linear(512*3, 512)
        self.classifier = torch.nn.Linear(512, 2)
    
    def forward(self, x, clinical):
        ncct, cta = x[:,:2,...],x[:,2:,...]
        return self.classifier(self.relu(self.feedforward(torch.cat([self.ncct_norm(self.ncct_imaging_backbone(ncct)), self.cta_norm(self.cta_imaging_backbone(cta)), self.clinical_norm(self.clinical_backbone(clinical))], dim=1))))

class SEResNext50MMLImaging(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ncct_imaging_backbone = SEResNext50(in_channels=1, num_classes=512, spatial_dims=3)
        self.cta_imaging_backbone = SEResNext50(in_channels=1, num_classes=512, spatial_dims=3)
        self.ncct_norm= Norm[Norm.BATCH, 1](num_features=512)
        self.cta_norm= Norm[Norm.BATCH, 1](num_features=512)
        self.relu = torch.nn.ReLU()
        self.feedforward = torch.nn.Linear(1024, 512)
        self.classifier = torch.nn.Linear(512, 1)
    
    def forward(self, ncct, cta):
        return self.classifier(self.relu(self.feedforward(torch.cat([self.ncct_norm(self.ncct_imaging_backbone(ncct)), self.cta_norm(self.cta_imaging_backbone(cta))], dim=1))))

class fuse_img_clinic(torch.nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch=1024, out_class=2, dropout=0.3, mode='cat'):
        super(fuse_img_clinic, self).__init__()

        self.mode = mode

        num_modal = len(in_ch)
        if self.mode == 'add':
            self.modality_weights = torch.nn.Parameter(torch.ones(num_modal))

        stages = []
        for in_c in in_ch:
            stages.append(torch.nn.Sequential(
                torch.nn.Linear(in_c, in_c),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(in_c, out_ch)
                ))
        self.stages = torch.nn.ModuleList(stages)

        self.classification = torch.nn.Sequential(
                            torch.nn.Linear(out_ch if self.mode == 'add' else out_ch*num_modal, out_ch),
                            torch.nn.Dropout(dropout),
                            torch.nn.Linear(out_ch, 1)
                            )

    def forward(self, modalities):

        if self.mode == 'add':
            features = 0.
        else:
            features = []

        for i, x in enumerate(modalities):

            x_ = self.stages[i](x)
            if self.mode == 'add':
                features += x_ * self.modality_weights[i]
            else:
                if i==0:
                    features = x_
                else:
                    features = torch.cat([features, x_ ], 1)

        out = self.classification(features)

        return out




def normalisation(in_ch, norm='batchnorm', group=32):
    if norm == 'instancenorm':
        norm = torch.nn.InstanceNorm3d(in_ch)
    elif norm == 'groupnorm':
        norm = torch.nn.GroupNorm(group, in_ch)
    elif norm == 'batchnorm':
        norm = torch.nn.BatchNorm3d(in_ch)
    elif callable(activation):
        norm = norm
    else:
        raise ValueError('normalisation type {} is not supported'.format(norm))
    return norm


def activation(act='relu'):
    if act == 'relu':
        a = torch.nn.ReLU(inplace=True)
    elif act == 'lrelu':
        a = torch.nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    else:
        raise ValueError('activation type {} is not supported'.format(act))
    return a


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class Flatten(torch.nn.Module):
    def forward(self, input):
        return torch.flatten(input, 1)
        # return input.view(input.size(0), -1)

class Predictor(torch.nn.Module):
    def __init__(self, indim, layer=4, meddim=None, outdim=None, usebn=True, lastusebn=False, usenoise=False, classification=None, drop=0.3):
        super(Predictor, self).__init__()
        self.usenoise = usenoise
        self.classification = classification
        if usenoise:
            indim = indim + indim // 2
        assert layer >= 1
        if meddim is None:
            meddim = indim
        if outdim is None:
            outdim = indim
        models = []
        for _ in range(layer - 1):
            models.append(torch.nn.Linear(indim, meddim))
            if usebn:
                models.append(torch.nn.BatchNorm1d(meddim))
            models.append(torch.nn.ReLU(inplace=True))
            indim = meddim
        models.append(torch.nn.Dropout(drop))
        models.append(torch.nn.Linear(meddim, outdim))
        if lastusebn:
            models.append(torch.nn.BatchNorm1d(outdim))
        self.model = torch.nn.Sequential(*models)
        if self.classification:
            self.cls_layer = torch.nn.Linear(outdim, self.classification)
    def forward(self, x):
        if self.usenoise:
            x = torch.cat(
                [x, torch.randn(x.size(0), x.size(1) // 2).to(x.device)], dim=1)
        x = self.model(x)
        if self.classification:
            out = self.cls_layer(x)
            return out, x
        return x

class MultiSwinTrans(torch.nn.Module):
    def __init__(self, clin_size=2, attention=False, follow=0, class_mode='cat'):
        super().__init__()

        #follow 0 only baseline, follow 1 baseline + f24h, follow 2 bl+f1w, follow 3 bl+f24h+f1w

        self.follow = follow
        self.clin_size = clin_size
        self.attention = attention
        in_ch = [768]

        patch_size = (16,16,16)
        window_size = ensure_tuple_rep(7, 3)
        self.pret_model = SwinViT(
            in_chans=1,
            embed_dim=48,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3
            )


        clin_feats = 768
        self.clin_fc = torch.nn.Linear(self.clin_size, clin_feats)

        in_ch.append(clin_feats)
        
        self.classify = fuse_img_clinic(in_ch, out_ch=512, out_class=2, dropout=0.3, mode=class_mode)
        
    def forward(self, x, c):
        feat_bl = self.pret_model(x)[4]
        feat_bl = rearrange(feat_bl, "n c h w d -> n 1 (c h w d)")
        feat_clin = self.clin_fc(c[:, :self.clin_size])
        out = self.classify([feat_bl.squeeze(1), feat_clin])
        
        return out


def init_model():
    return SEResNext50MMLALL()
    # models = {
    #     # "MLP": lambda: MLP(in_features=kwargs.get("in_features")),
    #     "SEResNext50Single": lambda: SEResNext50Single(),
    #     "SEResNext50MML": lambda: SEResNext50MML(),
    #     "SEResNext50MMLALL": lambda: SEResNext50MMLALL(),
    #     "SEResNext50MMLImaging": lambda: SEResNext50MMLImaging(),
    #     "MultiSwinTrans": lambda: MultiSwinTrans(),
    #     # "RandomForestClassifier": lambda: RandomForestClassifier(**kwargs),
    #     # "LogisticRegression": lambda: LogisticRegression(**kwargs)
    # }
    # return models[model]()

def load_pickled_model(checkpoint):
    with open(checkpoint, 'rb') as file:
        model = pickle.load(file)
    
    return model

if __name__ == "__main__":
    x = torch.randn(2, 4, 128, 128, 128)
    t = torch.randn(2, 2)
    model = init_model()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params/1e6:,}")
    out = model(x,t)
    print(out.shape)
