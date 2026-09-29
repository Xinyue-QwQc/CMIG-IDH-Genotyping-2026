import argparse
import os
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
from data.new_dataset import full_data

from torch.utils.data import DataLoader

from torch import nn

from sklearn.metrics import roc_auc_score,accuracy_score

from utils.tools import cal_params

# 导入baseline
from models.baseline.convnext import convnext_small
from models.baseline.DenseNet import generate_dense_model
from models.baseline.ResNet import generate_model
from models.baseline.ResNext import generate_model_ResNext
from models.baseline.SENet import senet3d50,senet3d101,senet3d152,senet3d200



local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()


# Basic Information
parser.add_argument('--user', default='cyx', type=str)

parser.add_argument('--experiment', default='baseline_full', type=str)

parser.add_argument('--model_name', default='efficientnet-b6', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='HDT_net,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
# have changed
parser.add_argument("--json_path",default="/homec/kuanghl2/Codes/data_information_final.json",type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--input_C', default=4, type=int)

parser.add_argument('--input_H', default=128, type=int)

parser.add_argument('--input_W', default=128, type=int)

parser.add_argument('--input_D', default=128, type=int)

# Training Information
parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_classes', default=2, type=int)

parser.add_argument('--seed', default=1234, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='1,2', type=str)

parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--batch_size', default=8, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=500, type=int)

parser.add_argument('--save_freq', default=100, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

parser.add_argument("--is_dist",default=True,type=bool)

args = parser.parse_args()


# 交叉熵损失
def idh_cross_entropy(input,target,weight):
    return cross_entropy(input,target,weight=weight,ignore_index=-1)


# 关注更难分类的部分
class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
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




def main_worker(model):

    if args.is_dist:
        # 分布式初始化
        torch.distributed.init_process_group('nccl')
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # 代表只存rank = 0的编号进程的内容
    if args.local_rank == 0:
        wandb.init(project=args.experiment, name=args.model_name)
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../template/log', args.experiment + args.model_name + args.date)


        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))


    if args.is_dist:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
    else:
        rank = args.local_rank


    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if args.is_dist:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(args.local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)
    else:
        model = nn.DataParallel(model).cuda()
    # param = model.parameters()
    param = list(filter(lambda p: p.requires_grad, model.parameters()))


    optimizer = torch.optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)


    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../HDT-Net/checkpoint',
                                      args.experiment + args.model_name + args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)



    resume = ""


    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        # checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    # 选择txt文件中文件   训练数据根目录
    train_set = full_data(args.json_path,mode="train")

    if args.is_dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        logging.info('Samples for train = {}'.format(len(train_set)))
        num_gpu = (len(args.gpu) + 1) // 2
        train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                                  drop_last=True, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers)

    valid_set = full_data(args.json_path,mode="valid")
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.info('Samples for valid = {}'.format(len(valid_set)))

    start_time = time.time()
    torch.set_grad_enabled(True)

    best_epoch = 0
    min_loss = 100.0

    for epoch in range(args.start_epoch, args.end_epoch):
        model.train()
        if args.is_dist:
            train_sampler.set_epoch(epoch)
            # setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        start_epoch = time.time()
        epoch_train_idh_loss = 0.0
        for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
            # optimizer.adjust_learning_rate(epoch, args.end_epoch, args.lr)

            optimizer.zero_grad()
            x, idh = data

            if args.is_dist:
                x = x.cuda(args.local_rank, non_blocking=True)
                idh = idh.cuda(args.local_rank, non_blocking=True)
                weight = torch.tensor([181, 937]).float().cuda(args.local_rank, non_blocking=True)
            else:
                x = x.cuda()
                idh = idh.cuda()
                weight = torch.tensor([181, 937]).float().cuda()

            idh_out = model(x)

            idh_loss = idh_focal_loss(idh_out,idh,weight)

            if args.is_dist:
                reduce_idh_loss = all_reduce_tensor(idh_loss, world_size=num_gpu).data.cpu().numpy()
                epoch_train_idh_loss += reduce_idh_loss / len(train_loader)
            else:
                epoch_train_idh_loss += idh_loss.item() / len(train_loader)

            if args.local_rank == 0:
                logging.info("Epoch:{} Iter:{} idh_loss: {:.5f}".format(epoch, i , reduce_idh_loss))

            idh_loss.backward()
            optimizer.step()

        idh_probs = []
        idh_class = []
        idh_target = []
        with torch.no_grad():
            model.eval()

            epoch_idh_loss = 0.0

            for i, data in enumerate(valid_loader):

                x, idh = data

                if args.is_dist:
                    x = x.cuda(args.local_rank, non_blocking=True)
                    idh = idh.cuda(args.local_rank, non_blocking=True)
                else:
                    x = x.cuda()
                    idh = idh.cuda()

                idh_out = model(x)
                idh_loss = idh_focal_loss(idh_out, idh, weight)
                epoch_idh_loss += idh_loss / len(valid_loader)
                idh_pred = F.softmax(idh_out, 1)

                idh_pred_class = torch.argmax(idh_pred, dim=1)

                idh_probs.append(idh_pred[0][1].cpu())

                idh_class.append(idh_pred_class.item())
                idh_target.append(idh.item())

            accuracy = accuracy_score(idh_target, idh_class)
            auc = roc_auc_score(idh_target, idh_probs)

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

                logging.info(
                    "Epoch:{}[best_epoch:{} | min_total_loss:{:.5f} | idh_loss:{:.5f} | idh_acc: {:.5f} | idh_auc:{:.5f}]"
                    .format(epoch,best_epoch,min_loss,epoch_idh_loss,accuracy,auc)
                )

        end_epoch = time.time()
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 3) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)
            # name value 横坐标
            wandb.log({'lr':optimizer.param_groups[0]['lr'],'idh_loss': epoch_train_idh_loss,'valid_idh_loss':epoch_idh_loss})
            # writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
            # writer.add_scalar('idh_loss:', epoch_train_idh_loss, epoch)
            # writer.add_scalar('valid_idh_loss:', epoch_idh_loss, epoch)

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch - start_epoch) / 60
            remaining_time_hour = (args.end_epoch - epoch - 1) * epoch_time_minute / 60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:

        wandb.finish()

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


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


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
    # logger.addHandler(ch)
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
        args.model_name = "efficientnet-b6"
        model = nets.EfficientNetBN("efficientnet-b6", spatial_dims=3, in_channels=4, num_classes=2)
    elif flag == 4:
        args.model_name = "convnext-small"
        model = convnext_small(num_classes=2,in_chans=4)
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
            args.model_name = "Senet50"
            model = senet3d50()

    if is_cal_param:
        print("The params is ",cal_params(model))

    return model


if __name__ == "__main__":
    # for i in range(6):
    model = select_model(1,is_cal_param=True)
    args.model_name = "focal_loss_" + args.model_name
    # print(args.model_name)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    if args.is_dist:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    # model = nets.EfficientNetBN("efficientnet-b6",spatial_dims=3,in_channels=4,num_classes=2)
    main_worker(model)
