import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

import random
import logging
import numpy as np
import time
# import setproctitle
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import monai.networks.nets as nets
import torch.distributed as dist
from torch.nn.functional import cross_entropy
import wandb
from models.baseline.FoundBioNet.FoundBioNetM import FoundBioNet
from DINOCls import DINOCls
from models.baseline.Mamba_vision3D import mamba_vision_T
from models.baseline.VoCoV2 import VoCoV2    
from models.baseline.ctnet_3d import CTNet3D
from models.baseline.multimodal_mrs import init_model
from models.baseline.Res3DNet import Residual_I3D
# from data.new_dataset import FN3D
# from data.raw_dataset import FN3D
from data.Combied_data import FN3D

# from data.reshaped_data import FN3D


from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator

from torch import nn

from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix

from utils.tools import cal_params
from utils.loss_function import Hybrid_loss

# 导入baseline
from models.baseline.convnext import convnext_tiny
from models.baseline.DenseNet import generate_dense_model
from models.baseline.ResNet import generate_model
from models.baseline.ResNext import generate_model_ResNext
from models.baseline.SENet import senet3d50,senet3d101,senet3d152,senet3d200
from models.baseline.convnextv2 import convnextv2_tiny
from models.baseline.efficient_netv2 import effnetv2_s
from models.baseline.StarNet import starnet_s4
from models.baseline.FasterNet import FasterNet
from models.baseline.TF_net import LightTF_Net
from models.baseline.M3D_VIT import M3D_VIT_CLS
from models.baseline.repvit import repvit_m1_5
from models.baseline.EVit import build_backbone
# from models.baseline.PFMLP import PFMLP
from models.baseline.WSOFNet import wsofNet
from models.baseline.AGGN import Aggn
from models.baseline.UnirepLKNet import unireplknet_t
from models.baseline.M3T_github import M3T
from models.baseline.BrainSegFounder import BrainSegFounder
from models.baseline.VoCo import VoCo
from torch.optim.lr_scheduler import CosineAnnealingLR
# from models.Revised_SAM_Med3D.encoder_for_cls import SAM_CLS
from models.Revised_SAM_Med3D.encoder_for_clsV3_ import SAM_CLS
# from models.Revised_SAM_Med3D.encoder_for_clsV3 import SAM_CLS
from models.Revised_SAM.encoder_for_cls import LVM_SAM_CLS
from models.baseline.MedViT3D import MedViT_small
from models.Language_model.BioLinkBert import BertModel
from models.Language_model.BioMedBert import MedBertModel
from models.Language_model.PMC_encoder import PMC_text_encoder
from transformers import AutoTokenizer


from test_net import test_main
# from models.SAM_Med3D.image_encoder3D import ImageEncoderViT3D



local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()


# Basic Information
parser.add_argument('--user', default='cyx', type=str)

parser.add_argument('--experiment', default='SAM_MED3D_LAB', type=str)

parser.add_argument('--model_name', default='efficientnet-b6', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='HDT_net,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
# have changed
# parser.add_argument("--json_path",default="/homec/kuanghl2/Codes/data_information_new.json",type=str)


parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--input_C', default=4, type=int)

parser.add_argument('--input_H', default=128, type=int)

parser.add_argument('--input_W', default=128, type=int)

parser.add_argument('--input_D', default=128, type=int)

# Training Information
parser.add_argument('--lr', default=0.0001, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_classes', default=2, type=int)

parser.add_argument('--seed', default=2356, type=int) # original is 1234, the candidate including 2356、3407 and so on 

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='1,3', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=500, type=int)

parser.add_argument('--save_freq', default=20, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

parser.add_argument("--is_dist",default=True,type=bool)

parser.add_argument("--train_root",default="/home/cyx/Datasets/Full_data/new_train",type=str)

parser.add_argument("--train_ls",default="/home/cyx/Datasets/Full_data/new_train/train_new.txt",type=str)

parser.add_argument("--valid_root",default="/home/cyx/Datasets/Full_data/new_valid",type=str)

parser.add_argument("--valid_ls",default="/home/cyx/Datasets/Full_data/new_valid/valid_new.txt",type=str)

parser.add_argument("--is_warmup",default=True,type=bool)

parser.add_argument("--warmup_epoch",default=20,type=int)

parser.add_argument("--data_index",default=0,type=int)

parser.add_argument("--data_json_path",default="",type=str)

parser.add_argument("--train_baseline",default=False,type=bool)

parser.add_argument("--selected_model_index",default=0,type=int)


args = parser.parse_args()



# 交叉熵损失
def idh_cross_entropy(input,target,weight):
    return cross_entropy(input, target,weight=weight,ignore_index=-1)



# 关注更难分类的部分
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5,reduction='mean'):
        # super(FocalLoss, self).__init__(weight,reduction=reduction)
        super().__init__()
        self.gamma = gamma
        # self.alpha = nn.Parameter(torch.ones(1))
        # self.alpha = 
        # self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        # 直接得到概率值
        ce_loss = F.cross_entropy(input, target,reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = 5*((1 - pt) ** self.gamma * ce_loss).mean() # self.alpha*
        return focal_loss



def idh_focal_loss(input,target,weight):
    focalloss = FocalLoss(weight=weight)

    return focalloss(input, target)



# 将损失分到其他GPU上，进行计算
def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor




def main_worker(model,loss_fn=idh_cross_entropy,use_brats=False,
                use_text=False, related_weights=None, data_json_path="/home/cyx/Datasets/Combied_Bra_UPENN_set.json"):

    # 分布式初始化
    torch.distributed.init_process_group('nccl')
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # 代表只存rank = 0的编号进程的内容
    if args.local_rank == 0:
        wandb.init(project=args.experiment, name=args.model_name)
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.model_name + args.date)
        log_file = os.path.join(log_dir,args.model_name + "_" + args.date + ".txt")
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))
        logging.info('----------------------------------------The model architecture is ----------------------------------')
        # print("The model architecture is:")
        print(model)
        # logging.info(model)


    torch.cuda.set_device(args.local_rank)
    rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if use_text:
        # tokenizer = AutoTokenizer.from_pretrained("/home/cyx/Datasets/HuggingFace_models/models--michiyasunaga--BioLinkBERT-large/")
        text_encoder = BertModel().cuda(args.local_rank)
        # tokenizer = AutoTokenizer.from_pretrained("/home/cyx/Datasets/HuggingFace_models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/")
        # text_encoder = PMC_text_encoder().cuda(args.local_rank)
        # tokenizer = AutoTokenizer.from_pretrained("/home/cyx/Datasets/HuggingFace_models/LLM_huggingface/medbert_pre/")
        # text_encoder = MedBertModel().cuda(args.local_rank)


    # model = class_model
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],
                                                   output_device=args.local_rank,
                                                   find_unused_parameters=True)
    
    # param = model.parameters()
    param = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    # lr_scheduler = CosineAnnealingLR(optimizer,T_max=(args.end_epoch - args.start_epoch), eta_min=0)


    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                      args.experiment + "_" + args.model_name + "_" + args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


    # resume = ""

    # writer = SummaryWriter()

    if os.path.isfile(args.resume) and args.load:
        logging.info('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        # checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    # 选择txt文件中文件   训练数据根目录
    train_set = FN3D(mode="train",use_addition=use_text,data_json_path=data_json_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    logging.info('Samples for train = {}'.format(len(train_set)))

    num_gpu = (len(args.gpu) + 1) // 2
    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    
    valid_set = FN3D(mode="valid",use_addition=use_text,data_json_path=data_json_path)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logging.info('Samples for valid = {}'.format(len(valid_set)))

    start_time = time.time()
    torch.set_grad_enabled(True)

    best_epoch = 0
    min_loss = 100.0
    best_acc = -1  
    best_acc_epoch = 0
    best_weighted_score = 0
    weighted_best_epoch = 0 

    if args.is_warmup:
        warmup_epoch = args.warmup_epoch
    else:
        warmup_epoch = 0

    for epoch in range(args.start_epoch, args.end_epoch):
        model.train()
        train_sampler.set_epoch(epoch)  # shuffle
        # setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        start_epoch = time.time()
        epoch_train_idh_loss = 0.0
        for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr,warmup_epoch=warmup_epoch)
            # optimizer.adjust_learning_rate(epoch, args.end_epoch, args.lr)

            optimizer.zero_grad()

            if use_text:
                x, idh, texts = data
                texts = text_encoder(texts)
            else:
                x, idh = data

            x = x.cuda(args.local_rank, non_blocking=True)
            idh = idh.cuda(args.local_rank, non_blocking=True)

            if use_brats:
                weight = torch.tensor(related_weights).float().cuda(args.local_rank, non_blocking=True)
            else:
                weight = torch.tensor([180, 565]).float().cuda(args.local_rank, non_blocking=True)

            if use_text:
                outs = model(x, texts)  # text_embed,t_g
                if isinstance(outs,tuple):
                    idh_out,text_loss = outs
                    idh_loss = loss_fn(idh_out,idh,weight)  + text_loss
                else:
                    idh_out = outs
                    idh_loss = loss_fn(idh_out,idh,weight)
            else:
                idh_out = model(x)

                idh_loss = loss_fn(idh_out,idh,weight)

            reduce_idh_loss = all_reduce_tensor(idh_loss, world_size=num_gpu).data.cpu().numpy()

            epoch_train_idh_loss += reduce_idh_loss / len(train_loader)

            if args.local_rank == 0:
                logging.info("Epoch:{} Iter:{} idh_loss: {:.5f}".format(epoch, i , reduce_idh_loss))
            idh_loss.backward()
            optimizer.step()

        # lr_scheduler.step()

        idh_probs = []
        idh_class = []
        idh_target = []
        with torch.no_grad():
            model.eval()

            epoch_idh_loss = 0.0

            for i, data in enumerate(valid_loader):

                if use_text:
                    x, idh, texts = data
                    texts = text_encoder(texts)

                else:
                    x, idh = data

                x = x.cuda(args.local_rank, non_blocking=True)

                idh = idh.cuda(args.local_rank, non_blocking=True)

                if use_text:
                    outs = model(x, texts) #text_embed,t_g
                    if isinstance(outs,tuple):
                        idh_out, text_loss = outs
                        idh_loss = loss_fn(idh_out, idh, weight) + text_loss
                    else:
                        idh_out = outs
                        idh_loss = loss_fn(idh_out, idh, weight)
                else:
                    idh_out = model(x)
                    idh_loss = loss_fn(idh_out, idh, weight)
                epoch_idh_loss += idh_loss / len(valid_loader)
                idh_pred = F.softmax(idh_out, 1)  # 8 * 2
                # size 8
                idh_pred_class = torch.argmax(idh_pred, dim=1)

                idh_probs += idh_pred[:,1].cpu().detach().numpy().tolist()

                # idh_probs.append(idh_pred[0][1].cpu())

                idh_class += idh_pred_class.cpu().detach().numpy().tolist()
                idh_target += idh.cpu().detach().numpy().tolist()

            accuracy = accuracy_score(idh_target, idh_class)
            confusion = confusion_matrix(idh_target, idh_class)
            
            auc = roc_auc_score(idh_target, idh_probs)


            specificity = 0
            if float(confusion[0, 0] + confusion[0, 1]) != 0:
                specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
            sensitivity = 0
            if float(confusion[1, 1] + confusion[1, 0]) != 0:
                sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
            p_spec = 0 
            if float(confusion[0, 0] + confusion[1, 0]) != 0:
                p_spec = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[1, 0])
            p_sens = 0
            if float(confusion[1, 1] + confusion[0, 1]) != 0:
                p_sens = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
            
            try:
                f1_spec = (2*specificity*p_spec)/(specificity+p_spec)
                f1_sens = (2*sensitivity*p_sens)/(sensitivity+p_sens)
            except:
                f1_spec = 0
                f1_sens = 0

            # 单独看有病的那一类
            weighted_score = f1_sens

            if args.local_rank == 0:

                if min_loss >= epoch_idh_loss:
                    min_loss = epoch_idh_loss
                    best_epoch = epoch
                    logging.info('there is an improvement that update the metrics and save the best model.')

                    file_name = os.path.join(checkpoint_dir, 'model_epoch_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)
                    
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_acc_epoch = epoch
                    file_name = os.path.join(checkpoint_dir, 'model_epoch_best_acc.pth')
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)
                
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    weighted_best_epoch = epoch
                    file_name = os.path.join(checkpoint_dir, 'model_epoch_best_weighted.pth')
                    # file_text_name = os.path.join(checkpoint_dir,"text")
                    # text_encoder.text_encoder.save_pretrained(file_text_name)
                    # file_name = os.path.join(checkpoint_dir, 'model_epoch_best_weighted.pth')
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)


                logging.info(
                    "Epoch:{}[lowest_loss_epoch:{} | best_acc_epoch:{} | best_weighted_epoch:{} | min_total_loss:{:.5f} | idh_loss:{:.5f} | idh_acc: {:.5f} | idh_auc:{:.5f}]"
                    .format(epoch,best_epoch,best_acc_epoch,weighted_best_epoch,min_loss,epoch_idh_loss,accuracy,auc)
                )

        end_epoch = time.time()
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_latest.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)
            # name value 横坐标
            wandb.log({'lr':optimizer.param_groups[0]['lr'],'idh_loss': epoch_train_idh_loss,
                       'valid_idh_loss':epoch_idh_loss,"valid_acc":accuracy,
                       
                       })
            epoch_time_minute = (end_epoch - start_epoch) / 60
            remaining_time_hour = (args.end_epoch - epoch - 1) * epoch_time_minute / 60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:

        wandb.finish()
        # print("The best acc is {}, and the corresponding epoch is {}".format(best_acc,best_acc_epoch))
        # print("The best weighted score is {}, and the corresponding epoch is {}".format(best_weighted_score,weighted_best_epoch))
        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'idh_state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9, warmup_epoch=0):
    # 进行warm up
    if epoch < warmup_epoch:
        for param_group in optimizer.param_groups:
            # Linear warmup strategy
            # param_group['lr'] = init_lr * 0.05 *(1+epoch)
            param_group['lr'] = init_lr * (epoch / warmup_epoch)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = round(init_lr * np.power(1 - (epoch-warmup_epoch) / (max_epoch-warmup_epoch), power), 8)


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


# 模型选择
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
        args.model_name = "efficientnetV2-s"
        model = effnetv2_s(num_classes=2)
    elif flag == 4:
        args.model_name = "convnext-tiny"
        model = convnext_tiny(num_classes=2,in_chans=4)
    elif flag == 5:
        args.model_name = "MedVIT"
        model = MedViT_small()
    elif flag == 6:
        args.model_name = "repvit_m1_5"
        model = repvit_m1_5(num_classes=2)
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

if __name__ == "__main__":
    # pass 
    # os.environ["PYTHONHASHSEED"] = str(args.seed)
    # model = select_model(0,is_cal_param=False) # depth=101,
    # # args.model_name = "Resnet50"
    # args.experiment = "baseline_set_data_reshape"
    # args.num_workers = 8
    # args.gpu = "1,2"
    use_brats = True
    args.is_warmup = False

    if args.train_baseline:
        use_text = False
    else:
        use_text = True

    data_index = args.data_index

    # data_split = {0 : "Bra", 1 : "UCSF", 2 : "Bra_UCSF"}
    data_split = {0: "No_ucsf_no_reduced", 1: "No_ucsf_reduced",
                  2: "no_brats_reduced_811",
                  3: "No_ucsf_reduced_60", 4: "No_ucsf_reduced_mask"}

    if args.train_baseline:
        args.experiment = "Baseline_Combined_data"
    else:
        args.experiment = 'SAM_MED3D_Combined'

    data_json_dic = {0:"/home/cyx/Codes/data_process/Data_json/data_dic.json",
                     1:"/home/cyx/Codes/data_process/Data_json/data_dic_reduce.json",
                     2:"/home/cyx/Codes/data_process/Data_json/no_brats.json",
                     3:"/home/cyx/Codes/data_process/Data_json/no_ucsf_reduced_60.json",
                     4:"/home/cyx/Codes/data_process/Data_json/data_dic_reduce_with_mask.json"}

    # # # after residual

    args.model_name = f"FinalVersion_+{data_split[data_index]}"

    # 分别代表 Brats UCSF  Brats+UCSF    426 [1,1]  [89, 565]  80, 496
    related_weights = [[89, 565],
                       [89, 426],
                       [113, 524],
                       [89, 378],
                       [87, 426]]


    args.data_json_path = data_json_dic[data_index]

    use_weights = related_weights[data_index]


    # 继续训练
    # args.resume = "/home/cyx/Codes/SAM_MED3D_for_CLS/checkpoint/SAM_MED3D_Combined_Discussion_RepVit+No_ucsf_no_reduced_2024-10-22/model_epoch_latest.pth"

    args.end_epoch = 300

    # args.model_name += "single"
    if args.train_baseline:
        # model = mamba_vision_T(num_classes=2,in_chans=4)
        # model = VoCoV2(ConvMoe=False,r=8)
        # model = SAM_CLS(conv_adapter=True, ConvMoe=True, mismatch=True, text_embed=True, reduction_ratio=2,
        #                 bg_know=True)
        # model = PFMLP()
        model = Residual_I3D()
        args.model_name = "Res3DNet"
    #     model = CTNet3D(
    #     in_ch_modality_a=2,   # CT
    #     in_ch_modality_b=2,   # MRI或其他模态
    #     num_classes=2,
    #     base_channels=64,     # 标准通道数
    #     embed_dim=96,
    #     dropout=0.5
    # )
    #     # model =  DINOCls(r=24,ConvMoe=True)
    #     args.model_name = "CTNet3D"
        # model = M3T()
        # model = starnet_s4(num_classes=2)
        # model = FasterNet(num_classes=2)
        # model = unireplknet_t()
        # model = wsofNet(in_chans=4,dim=384)
        # model = Aggn()
        # model = select_model(0)
        # model = M3D_VIT_CLS(in_chans=1, patch_size=[16, 8, 8], img_size=128,ConvMoe=True,text_prompt=True,select_features="patch",select_layer=1)
        # use_text = False
        # a = "Final_"
        # model = SAM_CLS(conv_adapter=False, ConvMoe=True, mismatch=False, text_embed=False)
        # model = repvit_m1_5(num_classes=2)
        # args.model_name = "UnirepLKNet_t"
        # args.model_name = "repvit_m1_5"
        # args.model_name = "FasterNet"
        # args.model_name = "WsoFNet"
        # args.model_name = "MambaVision_" + str(args.seed)
        # args.model_name = "VoCov2"
        # args.model_name = a + args.model_name
        # args.model_name = "StarNet_s4"
    else:
        use_text = True
        # model = DINOCls(multimodal_finetune=True,know=True,ConvMoe=True)
        # model = SAM_CLS(conv_adapter=True, ConvMoe=True, mismatch=True, text_embed=True,bg_know=True,in_channels=3)
        model = SAM_CLS(conv_adapter=True, ConvMoe=True, mismatch=True, text_embed=True,bg_know=True,in_channels=4)
        # model = init_model()
    # model = LVM_SAM_CLS(depth_hiden_channels=64) loss_fn=Hybrid_loss
    args.batch_size = 4

    # args.lr /= 2
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker(model,use_brats=use_brats,use_text=use_text,data_json_path=args.data_json_path,
                related_weights=use_weights)