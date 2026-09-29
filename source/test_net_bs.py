import os
import time
import random
import numpy as np
import wandb
from models.baseline.convnext import convnext_small
# from utils.net_util import tailor_and_concat
# import setproctitle
from monai.networks.nets import EfficientNetBN
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.baseline.TF_net import LightTF_Net
# from data.new_dataset import FN3D
# from data.raw_dataset import FN3D
from data.Combied_data import FN3D
import json
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
import pandas as pd
from utils.args import test_args

from models.Revised_SAM.encoder_for_cls import LVM_SAM_CLS
from models.baseline.convnext import convnext_tiny
from models.baseline.MedViT3D import MedViT_small
from models.baseline.MTDL import MTDL
from models.baseline.convnextv2 import convnextv2_tiny
from models.baseline.repvit import repvit_m1_5
from models.baseline.DenseNet import generate_dense_model
from models.baseline.ResNet import generate_model
from models.baseline.ResNext import generate_model_ResNext
from models.baseline.SENet import senet3d50,senet3d101,senet3d152,senet3d200
from models.baseline.efficient_netv2 import effnetv2_s
from models.baseline.UnirepLKNet import unireplknet_t
from models.baseline.StarNet import starnet_s4
from models.baseline.FasterNet import FasterNet
from models.baseline.M3D_VIT import M3D_VIT_CLS
from models.baseline.M3T_github import M3T
from models.baseline.Mamba_vision3D import mamba_vision_T
# from models.Revised_SAM_Med3D.encoder_for_cls import SAM_CLS
from models.Revised_SAM_Med3D.encoder_for_clsV3_ import SAM_CLS
from MambaIDH import MambaIDH_T
# from models.SAM_Med3D.image_encoder3D import ImageEncoderViT3D

from utils.tools import cal_params
from monai.networks import nets

from models.Language_model.BioLinkBert import BertModel
from transformers import AutoTokenizer

def tailor_and_concat(x, model,age,gender):
    temp = []
    idh_temp=[]
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
        idh_out = model(temp[i],age,gender)
        # grade_out = model['grade'](encoder_outs[3], encoder_outs[4])
        # print("idh_out:",idh_out)
        idh_temp.append(idh_out)

    idh_out = torch.mean(torch.stack(idh_temp), dim=0)
    # print("idh_out mean:",idh_out)
    return idh_out#,grade_out


def test_main(model,args,load_file,use_brats=False,use_text=False,TTA=False,data_json_path=""):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    IDH_model = model
    IDH_model = torch.nn.DataParallel(IDH_model).cuda()
    wandb.init(project=args.experiment,name=args.model_name)
    print("The model architecture is:")
    print(model)



    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        check = checkpoint["state_dict"]
        # for k in list(check.keys()):
        #     if "after_confusion" in k:
        #         continue
        #     if "cls_net" in k:
        #         continue
        #     if "conv_adapter1" in k:
        #         continue
        #     del check[k]
        IDH_model.load_state_dict(check)
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(load_file))
    else:
        print('There is no resume file to load!')

    valid_set = FN3D(mode='test',use_addition=use_text,data_json_path=data_json_path,use_bs=True)
    record = OrderedDict()
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # print("valid_loader",valid_loader)

    start_time = time.time()

    if use_text:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/cyx/Datasets/HuggingFace_models/models--michiyasunaga--BioLinkBERT-large/")
        text_encoder = BertModel().cuda()

    with torch.no_grad():

        IDH_model.eval()
        idh_prob = []
        idh_class = []
        idh_truth = []
        idh_error_case = []
        ids = []
        names = valid_set.names

        for i, data in enumerate(valid_loader):
            print('-------------------------------------------------------------------')
            msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))

            # print("data[0]:", data[0].shape, 'data[1]', data[1])
            # data = [t.cuda(non_blocking=True) for t in data]
            if use_text:
                x, idh, gender, age = data

                if len(age.shape) == 1:
                    age = age.unsqueeze(1)
                if len(gender.shape) == 1:
                    gender = gender.unsqueeze(1)

                # age = age.cuda(args.local_rank, non_blocking=True)
                # gender = gender.cuda(args.local_rank, non_blocking=True)
            else:
                x, idh = data

            # x, idh = data[:2]
            # x = x[..., :155]
            # 测试时增强
            if use_text :
                pred = tailor_and_concat(x, IDH_model, age,gender)
            else:
                pred = tailor_and_concat(x, IDH_model)

            idh_pred = F.softmax(pred, 1)


            # idh_pred = F.softmax(pred, 1)

            # print("idh_pred:", idh_pred)
            idh_prob.append(idh_pred[0][1].item())

            idh_pred_class = torch.argmax(idh_pred, dim=1)

            idh_class.append(idh_pred_class.item())
            # print('id:', names[i], 'IDH_truth:', idh.item(), 'IDH_pred:', idh_pred_class.item())

            ids.append(names[i])
            idh_truth.append(idh.item())
            if not (idh_pred_class.item() == idh.item()):
                idh_error_case.append({'id': names[i], 'truth:': idh.item(), 'pred': idh_pred_class.item()})

            name = str(i)
            if names:
                name = names[i]
                msg += '{:>20}, '.format(name)

            print(msg)

        print("--------------------------------IDH evaluation report---------------------------------------")

        data = pd.DataFrame({"ID": ids, "pred": idh_prob, "pred_class": idh_class, "idh_truth": idh_truth})

        csv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"csv_record",args.model_name)

        if not os.path.exists(csv_path):
            os.makedirs(csv_path)

        data.to_csv(csv_path + "/predict.csv")

        confusion = confusion_matrix(idh_truth, idh_class)
        print(confusion)
        labels = [0, 1]
        target_names = ["wild", "Mutant"]

        auc = roc_auc_score(idh_truth, idh_prob)
        acc = accuracy_score(idh_truth, idh_class)

        record["AUC"] = auc
        record["ACC"] = acc
        record["Confusin Matrix"] = {"TP": str(confusion[0, 0]), "FN": str(confusion[0, 1]), "FP": str(confusion[1, 0]),
                                        "TN": str(confusion[1, 1])}
        record["Description"] = "Sensitivity is mutant, Specificity is wild. 0 is wild, 1 is mutant. "


        print(classification_report(idh_truth, idh_class, labels=labels, target_names=target_names))
        print("AUC:", auc)
        print("Acc:", acc)
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        print("Global Accuracy: " + str(accuracy))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        print("Specificity: " + str(specificity))
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        print("Sensitivity: " + str(sensitivity))
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        print("Precision: " + str(precision))
        print("-------------------------- error cases----------------------------------------")
        record["Global Accuracy"] = accuracy
        record["Specificity"] = specificity
        record["Sensitivity"] = sensitivity
        record["Precision"] = precision
        record["Error Case"] = idh_error_case
        json_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"result",args.model_name)
        wandb.log({"AUC":auc,"Acc":acc})
        wandb.log({"Specificity":specificity,"Sensitivity":sensitivity})
        for case in idh_error_case:
            print(case)

        end_time = time.time()
        full_test_time = (end_time - start_time) / 60
        average_time = full_test_time / len(valid_set)
        print('{:.2f} minutes!'.format(average_time))

        record["Spend"] = f"The average time is {average_time} minutes."

        if not os.path.exists(json_path):
            os.makedirs(json_path)

        # if pt < 10:
        #     json_name = str(pt)+"_init_result.json"
        # else:
        json_name = args.json_name

        with open(os.path.join(json_path, json_name), "w") as f:
            json.dump(record, f, indent=4, sort_keys=True)


def select_model(flag,depth=None,is_cal_param=False):

    if flag == 0:
        if depth:
            args.model_name = "Densenet" + str(depth)
            model = generate_dense_model(depth)

        else:
            args.model_name = "Densenet121"
            model = generate_dense_model(121)

    elif flag == 1:

        if depth:
            args.model_name = "Resnet" + str(depth)
            model = generate_model(depth)
        else:
            args.model_name = "Resnet50"
            model = generate_model(50)
    elif flag == 2:
        if depth:
            args.model_name = "Resnext" + str(depth)
            model = generate_model_ResNext(depth)
        else:
            args.model_name = "Resnext50"
            model = generate_model_ResNext(50)
    elif flag == 3:
        args.model_name = "efficientnetv2-s"
        model = effnetv2_s(num_classes=2)
    elif flag == 4:
        args.model_name = "convnext-small"
        model = convnext_tiny(num_classes=2,in_chans=4)
    elif flag == 5:
        args.model_name = "MedVIT"
        model = MedViT_small()
    else:
        if depth:
            assert depth in [50,101,152,200]
            args.model_name = "Senet" + str(depth)
            if depth == 50:
                model = senet3d50()
            elif depth == 101:
                model = senet3d101()
            elif depth == 152:
                model = senet3d152()
            else:
                model = senet3d200()
        else:
            args.model_name = "Senet101"
            model = senet3d101()

    if is_cal_param:
        print("The params is ",cal_params(model))

    return model



if __name__ == '__main__':
    args = test_args(model_name="Senet50_lowest_loss",experiment="baseline_part",gpu="2")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 两者默认都为False
    args.external = True
    args.baseline = True

    if not args.external:
        args.experiment = "SAM_MED_Combined_Test"
    else:
        args.experiment = "SAM_Med_External_Test"

    if args.baseline:
        args.experiment = "Combined_Baseline_Test"
        # args.experiment = "Mamba_experiments_test"


    data_index = 0


    data_split = {0: "No_ucsf_no_reduced", 1: "No_ucsf_reduced", 2: "no_brats_reduced_811", 3: "No_ucsf_reduced_60",
                  4: "No_ucsf_reduced_mask"
                  }



    data_json_dic = {0:"/home/cyx/Codes/data_process/Data_json/data_dic.json",
                     1:"/home/cyx/Codes/data_process/Data_json/data_dic_reduce.json",
                     2:"/home/cyx/Codes/data_process/Data_json/no_brats.json",
                     3:"/home/cyx/Codes/data_process/Data_json/no_ucsf_reduced_60.json",
                     4:"/home/cyx/Codes/data_process/Data_json/data_dic_reduce_with_mask.json"}

    # after residual

    # after residual

    args.model_name = f"Ablation_Block_no_Feat+{data_split[data_index]}"

    if args.external:
        data_json_dic = {
            0: "/home/cyx/Codes/data_process/Data_json/3d_path_UCSF_set_external.json",
            1: "/home/cyx/Codes/data_process/Data_json/3d_path_UCSF_set_external.json",
            2: "/home/cyx/Codes/data_process/Data_json/3d_path_Bra_set_external.json",
            3: "/home/cyx/Codes/data_process/Data_json/3d_path_Bra_set_external.json",
            4: "/home/cyx/Codes/data_process/Data_json/3d_path_UCSF_set_external.json",
        }

    use_text = True
    # model = select_model(flag=0)
    if not args.baseline:

        # 得到分类网络
        model = SAM_CLS(conv_adapter=True,ConvMoe=True,mismatch=True,text_embed=use_text,reduction_ratio=2)
    else:

        # model = M3T()
        # model = starnet_s4(num_classes=2)
        # model = mamba_vision_T(num_classes=2,in_chans=4)
        # model = unireplknet_t()
        # model = MambaIDH_T(num_classes=2, in_chans=4)
        model = MTDL(in_chans=4,num_classes=2)
        # model = select_model(3)
        # model = repvit_m1_5(num_classes=2)
        # model = LightTF_Net(in_chans=4,make_cls=True)
        # args.model_name = "unireplknet_t"
        # model = M3D_VIT_CLS(in_chans=1, patch_size=[16, 8, 8], img_size=128,ConvMoe=False,text_prompt=True)
        # args.model_name = "FasterNet"
        args.model_name = "MTDL"
        # args.model_name = "repvit_m1_5"
        # args.model_name = "MambaIDH_T"


        # model = M3D_VIT_CLS(in_chans=1, patch_size=[16, 8, 8], img_size=128,ConvMoe=True,text_prompt=True,select_features="patch",select_layer=1)
        # use_text = True


    if args.external:
        args.model_name += "_External"


    test_main(model,args,use_brats=False,use_text=use_text,data_json_path=data_json_dic[data_index],
         load_file="/home/cyx/Codes/SAM_MED3D_for_CLS/checkpoint/Baseline_Combined_data_Final_MTDL__2024-09-11/model_epoch_best.pth")

    