import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from visualizer import get_local
get_local.activate()
from models.Revised_SAM_Med3D.encoder_for_clsV3_ import SAM_CLS
from models.baseline.BrainSegFounder import BrainSegFounder
# from models.baseline.convnext import convnext_tiny
# from models.baseline.MedViT3D import MedViT_small
# from models.baseline.convnextv2 import convnextv2_tiny
from models.baseline.repvit import repvit_m1_5
# from models.baseline.DenseNet import generate_dense_model
# from models.baseline.ResNet import generate_model
# from models.baseline.ResNext import generate_model_ResNext
# from models.baseline.SENet import senet3d50,senet3d101,senet3d152,senet3d200
# from models.baseline.efficient_netv2 import effnetv2_s
# from models.baseline.UnirepLKNet import unireplknet_t
# from models.baseline.StarNet import starnet_s4
# from models.baseline.FasterNet import FasterNet
# from models.baseline.M3D_VIT import M3D_VIT_CLS
# from models.baseline.M3T_github import M3T
# from models.baseline.Mamba_vision3D import mamba_vision_T
from models.baseline.WSOFNet import wsofNet
from models.baseline.VoCoV2 import VoCoV2
from models.baseline.EVit import build_backbone
from models.baseline.ctnet_3d import CTNet3D
from models.baseline.multimodal_mrs import init_model
from DINOCls import DINOCls

import matplotlib.pyplot as plt
import torch.nn as nn
from typing import Optional
import torch
import numpy as np
import json
from models.Language_model.BioLinkBert import BertModel
from transformers import AutoTokenizer


def load_json(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data


def tailor_and_concat(x, model,text_embed:Optional[torch.Tensor]=None,tg:Optional[torch.Tensor]=None,name="ErrorCase",save_path="./MMAN_Feature"):

    save_path = os.path.join(save_path,name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    temp = []
    idh_temp=[]
    tSNE_ls = []

    # 图像裁剪为八块
    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    for i in range(len(temp)):
        if text_embed is not None:
            out = model(temp[i],text_embed)
            if len(out) == 2:
                idh_out = out[0]
            else:
                idh_out = out
        else:
            idh_out = model(temp[i])

        if torch.any(torch.isnan(idh_out)):
            continue

        attn_maps = get_local.cache
        ## attn_map
        for k,v in attn_maps.items():
            for j,attn_map in enumerate(v):
                np.save(os.path.join(save_path,f"{i}_{j}.npy"),attn_map)   # i代表哪一块 j代表哪一层
        
        get_local.clear()

        # tSNE = idh_out.detach().cpu().numpy()

    #     tSNE_dic = get_local.cache
    #     for k,v in tSNE_dic.items():
    #         tSNE = v[0]
    #     if np.any(np.isnan(tSNE)):
    #         get_local.clear()
    #         continue
        # tSNE_ls.append(tSNE)
    #     get_local.clear()
    #
    #     idh_temp.append(idh_out)
    #
    # # idh_out = torch.mean(torch.stack(idh_temp), dim=0)
    # tSNE_final = np.stack(tSNE_ls,axis=0)[:,0,:]
    # tSNE_final = np.max(tSNE_final,axis=0,keepdims=True)
    # return tSNE_final
    return 0

def preprocess(path,addition=False,age_gender_info="/home/cyx/Datasets/All_with_age_and_gender2.json",
               labeled_json="/home/cyx/Datasets/labeled_data.json",language_t=0):

    if addition:
        patient_info = load_json(age_gender_info)

    age_genders = []

    label = load_json(labeled_json)

    name = path.split("/")[-1][:-4]
    idh = label[name]

    data = np.load(path)

    if addition:
        age_genders= patient_info[name]
        langauge_template = ["The {} brain glioma patient is {} years old.",
                             "The {} patient is {} years old.",
                              "{} is {} years old and suffers from brain glioma."]
        text_prompt = langauge_template[language_t].format(age_genders[0], int(age_genders[1]))
        if age_genders[0] == "Male":
                age_genders[0] = 0
        else:
            age_genders[0] = 1
        # print()
        # gender,age = age_gender[0],age_gender[1]

        # pt_info = torch.tensor(age_genders,dtype=torch.float).unsqueeze(0)


    image = data
    image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')

    image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
    image = torch.from_numpy(image).float()

    images = image.unsqueeze(0)
    label = torch.tensor(idh)

    if addition:
        return images,label, text_prompt, name
    else:
        return images,label,name


def acquire_attn_map(paths, check_path, model, addition=False,model_name="Efficient_net"):

    model = torch.nn.DataParallel(model).cuda()
    print("The model architecture is as follow:")
    print(model)

    if os.path.exists(check_path):
        checkpoint = torch.load(check_path)
        check = checkpoint["state_dict"]
        model.load_state_dict(check)
        print('Successfully load checkpoint {}'.format(check_path))
    else:
        print("Error, the checkpoint not exist.")
        return 0

    if addition:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/cyx/Datasets/HuggingFace_models/models--michiyasunaga--BioLinkBERT-large/")
        text_encoder = BertModel().cuda()

    feature_ls = []
    with torch.no_grad():
        model.eval()
        for path in paths:
            if addition:
                data, label, text_embed,name = preprocess(path,addition)
                # text_embed = tokenizer(text_embed, return_tensors="pt", padding=True, truncation=True, max_length=20)
                # text_embed = text_embed.to("cuda")
                text_embed = text_encoder(text_embed)
            else:
                data, label, name = preprocess(path, addition)

            data = data.cuda()

            if addition:
                feature = tailor_and_concat(data, model, text_embed)
            else:
                feature = tailor_and_concat(data, model)

            feature_ls.append(feature)
    features = np.stack(feature_ls,axis=0)[:,0,:]
    np.save(os.path.join(f"/home/cyx/Codes/SAM_MED3D_for_CLS/TestSetTensor/{model_name}.npy"),features)

if __name__ == "__main__":
    # data_json = load_json("/home/cyx/Codes/data_process/Data_json/data_dic.json")
    # test_set_json = data_json["Test"]
    # paths = test_set_json
    paths = ["/home/cyx/Datasets/Full_data/train/UPENN-GBM-00585_11.npy"]
    check_path = "/home/cyx/Codes/SAM_MED3D_for_CLS/checkpoint/SAM_MED3D_Combined_Ablation_convlora+No_ucsf_no_reduced_2024-09-03/model_epoch_best_weighted.pth"
    # model = SAM_CLS(conv_adapter=True, ConvMoe=True, mismatch=True, text_embed=True,reduction_ratio=2,bg_know=True)
    # model =  DINOCls(ConvMoe=False)
    model = SAM_CLS(conv_adapter=True, ConvMoe=True, mismatch=True, text_embed=True,bg_know=False,in_channels=4)

    # model = repvit_m1_5(num_classes=2)
    # model = wsofNet(in_chans=4,dim=384)
    # model = BrainSegFounder()
    acquire_attn_map(paths,check_path,model,addition=True,model_name="Multimodal_MRS")
