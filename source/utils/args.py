import argparse
import time

def train_args(lr=2e-4, input_C=4,
               num_classes=2,weight_decay=1e-5,
               gpu="1,3", batch_size=4, start_epoch=0,end_epoch=300
               ,save_fre=50, experiment="baseline",
               model_name="ConvNext_small",
               is_dist=True, use_amp=False,num_workers=8):
    """
    :param lr:
    :param input_C:
    :param num_classes:
    :param weight_decay:
    :param gpu:
    :param batch_size:
    :param start_epoch:
    :param end_epoch:
    :param save_fre:
    :param experiment:
    :param model_name:
    :param is_dist:
    :param use_amp:
    :param num_workers:
    :return:
    """

    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parser = argparse.ArgumentParser()

    # Basic Information
    parser.add_argument('--user', default='cyx', type=str)

    parser.add_argument('--experiment', default=experiment, type=str)

    parser.add_argument('--model_name', default=model_name, type=str)

    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

    parser.add_argument('--description',
                        default='model,'
                                'training on train.txt!',
                        type=str)

    # DataSet Information
    # have changed
    parser.add_argument('--root', default='/home/cyx/Datasets/BraData/', type=str)

    parser.add_argument('--train_dir', default='Train', type=str)

    parser.add_argument('--valid_dir', default='Train', type=str)

    parser.add_argument('--test_dir', default='Val', type=str)

    parser.add_argument('--mode', default='train', type=str)

    parser.add_argument('--train_file', default='IDH_train_1.txt', type=str)  # IDH_all.txt

    parser.add_argument('--valid_file', default='IDH_valid_1.txt', type=str)  # IDH_test.txt

    parser.add_argument('--test_file', default='IDH_test.txt', type=str)

    parser.add_argument('--input_C', default=input_C, type=int)

    parser.add_argument("--num_classes", default=num_classes, type=int)

    # Training Information
    parser.add_argument('--lr', default=lr, type=float)

    parser.add_argument('--weight_decay', default=weight_decay, type=float)

    parser.add_argument('--amsgrad', default=True, type=bool)

    parser.add_argument('--seed', default=1234, type=int)

    parser.add_argument('--no_cuda', default=False, type=bool)

    parser.add_argument('--gpu', default=gpu, type=str)

    parser.add_argument('--num_workers', default=num_workers, type=int)

    parser.add_argument('--batch_size', default=batch_size, type=int)

    parser.add_argument('--start_epoch', default=start_epoch, type=int)

    parser.add_argument('--end_epoch', default=end_epoch, type=int)

    parser.add_argument('--save_freq', default=save_fre, type=int)

    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--load', default=True, type=bool)

    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    # parser.add_argument("--sam_med_check",default="/home/kuanghl/Codes/cyx/CODES/GIT_Project/sam-med2d_b.pth",type=str)

    parser.add_argument("--is_dist", default=is_dist, type=bool)

    parser.add_argument("--use_amp", default=use_amp, type=bool)

    args = parser.parse_args()

    return args

def test_args(experiment='baseline'
              ,test_date='2023-09-08',
              test_file="model_epoch_best.pth",
              input_C=4,num_workers=4,num_classes=2,
              gpu="2",
              model_name='ConvNext_small_no_amp'):
    """
    :param experiment:
    :param test_date:
    :param test_file:
    :param input_C:
    :param num_workers:
    :param num_classes:
    :param gpu:
    :param model_name:
    :return:
    """

    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parser = argparse.ArgumentParser()

    parser.add_argument('--user', default='name of user', type=str)

    parser.add_argument('--root', default='/home/cyx/Datasets/BraData/', type=str)

    parser.add_argument('--valid_dir', default='Val', type=str)

    parser.add_argument('--valid_file', default='IDH_test.txt', type=str)

    parser.add_argument('--output_dir', default='output', type=str)

    parser.add_argument('--submission', default='submission', type=str)

    parser.add_argument('--visual', default='visualization', type=str)

    parser.add_argument('--experiment', default=experiment+"_test_result", type=str)

    parser.add_argument("--check_experiment",default=experiment,type=str)

    parser.add_argument('--model_name', default=model_name, type=str)

    parser.add_argument('--test_date', default=test_date, type=str)

    parser.add_argument('--test_file', default=test_file,type=str)

    parser.add_argument('--use_TTA', default=True, type=bool)

    parser.add_argument('--post_process', default=True, type=bool)

    parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

    parser.add_argument("--input_C", default=input_C, type=int)

    parser.add_argument('--crop_H', default=128, type=int)

    parser.add_argument('--crop_W', default=128, type=int)

    parser.add_argument('--crop_D', default=128, type=int)

    parser.add_argument('--seed', default=1000, type=int)

    parser.add_argument('--num_classes', default=num_classes, type=int)

    parser.add_argument('--no_cuda', default=False, type=bool)

    parser.add_argument('--gpu', default=gpu, type=str)

    parser.add_argument('--num_workers', default=num_workers, type=int)

    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

    parser.add_argument("--json_name",default="best_epoch_result.json",type=str)

    parser.add_argument('--external', default=False, type=bool)

    parser.add_argument("--baseline",default=False,type=bool)

    args = parser.parse_args()

    return args