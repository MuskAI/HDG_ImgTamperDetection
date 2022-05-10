import torch
import torch.optim as optim
import torch.utils.data.dataloader
import os, sys
sys.path.append('')
sys.path.append('../utils')
import argparse
import time, datetime
from functions import my_f1_score, my_acc_score, my_precision_score, cross_entropy_loss_edge_weight, wce_huber_loss, \
    wce_huber_loss_8, my_recall_score, cross_entropy_loss, wce_dice_huber_loss,CE_loss,three_branch_loss
from torch.nn import init
from datasets.dataloader import TamperDataset
from PIL import Image
import shutil
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from utils import Logger, Averagvalue, weights_init, load_pretrained
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
# from model.unet_model import UNet as Net
from model.unet_model_aspp_attention import UNet as Net
import wandb

"""
Created by HDG
time: 2022/01/6
description:
1. finetune篡改区域
2. 画图
"""

""""""""""""""""""""""""""""""
"          参数               "
""""""""""""""""""""""""""""""

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=40, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--model_save_dir', type=str, help='model_save_dir',
                    default='save_model/aspp_atention_model')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=20, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
#####################resume##########################
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--per_epoch_freq', type=int, help='per_epoch_freq', default=50)

parser.add_argument('--fuse_loss_weight', type=int, help='fuse_loss_weight', default=12)
# ================ dataset

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

""""""""""""""""""""""""""""""
"          路径               "
""""""""""""""""""""""""""""""
model_save_dir = abspath(dirname(__file__))
model_save_dir = join(model_save_dir, args.model_save_dir)
if not isdir(model_save_dir):
    os.makedirs(model_save_dir)

""""""""""""""""""""""""""""""
"    ↓↓↓↓需要修改的参数↓↓↓↓     "
""""""""""""""""""""""""""""""

# wandb 使用

wandb.init(project="HDG_tamper", entity="muskai")
wandb.config = {
  "learning_rate": args.lr,
  "epochs": args.maxepoch,
  "batch_size": args.batch_size
}
wandb.watch_called = True
output_name_file_name = '0404-%d-%f-[f1%f-precision%f-acc%f-recall%f]-' \
                        '[f1%f-precision%f-acc%f-recall%f]-' \
                        '[f1%f-precision%f-acc%f-recall%f].pth'
""""""""""""""""""""""""""""""
"    ↑↑↑↑需要修改的参数↑↑↑↑     "
""""""""""""""""""""""""""""""








""""""""""""""""""""""""""""""
"          程序入口            "
""""""""""""""""""""""""""""""


def main():
    args.cuda = True
    # 1 choose the data you want to use
    using_data = {'my_sp': True,
                  'my_cm': True,
                  'template_casia_casia': True,
                  'template_coco_casia': True,
                  'cod10k': True,
                  'casia': False,
                  'coverage': False,
                  'columb': False,
                  'negative': True,
                  'negative_casia': False,
                  'texture_sp': True,
                  'texture_cm': False,
                  }

    # 2 define 3 types
    trainData = TamperDataset(using_data=using_data, train_val_test_mode='train')
    valData = TamperDataset( using_data=using_data, train_val_test_mode='val')

    # 3 specific dataloader
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                                  pin_memory=True)
    valDataLoader = torch.utils.data.DataLoader(valData, batch_size=args.batch_size, num_workers=4)

    # model
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    model.apply(weights_init)
    # 模型初始化
    # 如果没有这一步会根据正态分布自动初始化
    # model.apply(weights_init)

    # 模型可持续化

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'".format(args.resume))
            # optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            print("=> 想要使用预训练模型，但是路径出错 '{}'".format(args.resume))
            sys.exit(1)

    else:
        print("=> 不使用预训练模型，直接开始训练 '{}'".format(args.resume))

    # 调整学习率
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    # 数据迭代器

    for epoch in range(args.start_epoch, args.maxepoch):
        train_avg = train(model=model, optimizer=optimizer, dataParser=trainDataLoader, epoch=epoch)
        val_avg = val(model=model, dataParser=valDataLoader, epoch=epoch)

        """"""""""""""""""""""""""""""
        "          写入图            "
        """"""""""""""""""""""""""""""
        try:
            pass
        except Exception as e:
            print(e)

        """"""""""""""""""""""""""""""
        "          写入图            "
        """"""""""""""""""""""""""""""

        # 保存模型

        output_name = output_name_file_name % \
                      (epoch, train_avg['loss_avg'],

                       val_avg['f1_avg'],
                       val_avg['precision_avg'],
                       val_avg['accuracy_avg'],
                       val_avg['recall_avg'],


                       val_avg['f1_band_avg'],
                       val_avg['precision_band_avg'],
                       val_avg['accuracy_band_avg'],
                       val_avg['recall_band_avg'],


                       val_avg['f1_band_avg'],
                       val_avg['precision_band_avg'],
                       val_avg['accuracy_band_avg'],
                       val_avg['recall_band_avg']


                       )
        if epoch % 10 == 0:
            save_model_name = os.path.join(args.model_save_dir, output_name)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       save_model_name)

        scheduler.step(epoch=epoch)

    print('训练已完成!')


""""""""""""""""""""""""""""""
"           训练              "
""""""""""""""""""""""""""""""


def train(model, optimizer, dataParser, epoch):
    # 读取数据的迭代器

    train_epoch = len(dataParser)
    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    losses_area = Averagvalue()
    losses_band = Averagvalue()
    losses_cls = Averagvalue()

    # switch to train mode
    model.train()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        # 准备输入数据
        images = input_data['tamper_image'].cuda()
        labels = input_data['gt'].cuda()
        gt_band = input_data['gt_band'].cuda()
        gt_cls = input_data['gt_cls'].cuda()

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
        else:
            loss = torch.zeros(1)

        with torch.set_grad_enabled(True):
            images.requires_grad = True
            optimizer.zero_grad()
            # 网络输出
            output_area = model(images)

            """"""""""""""""""""""""""""""
            "         Loss 函数           "
            """"""""""""""""""""""""""""""

            loss = cross_entropy_loss(output_area, labels)
            # 记录总loss值
            wandb.log({
                'loss': loss.item()
            })
            loss.backward()
            optimizer.step()




        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        torch.cuda.empty_cache()

        batch_time.update(time.time() - end)
        end = time.time()


        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, train_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses)+ \
                   'Area Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_area) +\
                   'Band Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_band)+ \
                   'cls Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses_cls)


            print(info)

        if batch_index >= train_epoch:
            break

    return {'loss_avg': losses.avg}


@torch.no_grad()
def val(model, dataParser, epoch):
    # 读取数据的迭代器
    val_epoch = len(dataParser)

    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    f1_value = Averagvalue()
    acc_value = Averagvalue()
    recall_value = Averagvalue()
    precision_value = Averagvalue()

    f1_value_band = Averagvalue()
    acc_value_band = Averagvalue()
    recall_value_band = Averagvalue()
    precision_value_band = Averagvalue()
    f1_value_cls = Averagvalue()
    acc_value_cls = Averagvalue()
    recall_value_cls = Averagvalue()
    precision_value_cls = Averagvalue()

    # switch to test mode
    model.eval()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)

        images = input_data['tamper_image']
        labels = input_data['gt']
        gt_band = input_data['gt_band']
        gt_cls = input_data['gt_cls']
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
        else:
            loss = torch.zeros(1)

        # 网络输出
        output_area= model(images)

        """"""""""""""""""""""""""""""
        "         Loss 函数           "
        """"""""""""""""""""""""""""""


        # 将各种数据记录到专门的对象中

        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标

        f1score = my_f1_score(output_area, labels)
        precisionscore = my_precision_score(output_area, labels)
        accscore = my_acc_score(output_area, labels)
        recallscore = my_recall_score(output_area, labels)

        # print(output_band)
        # print(gt_band)
        # f1score_band = my_f1_score(output_band, gt_band)
        # precisionscore_band = my_precision_score(output_band, gt_band)
        # accscore_band = my_acc_score(output_band, gt_band)
        # recallscore_band  = my_recall_score(output_band, gt_band)
        #
        # f1score_cls = my_f1_score(output_cls, gt_cls)
        # precisionscore_cls = my_precision_score(output_cls, gt_cls)
        # accscore_cls = my_acc_score(output_cls, gt_cls)
        # recallscore_cls = my_recall_score(output_cls, gt_cls)


        # 记录实验数据
        wandb.log({
            'f1': f1score,
            'precision': precisionscore,
            'accuracy': accscore,
            'recall': recallscore
        })
        # writer.add_scalar('val_f1_score', f1score, global_step=epoch * val_epoch + batch_index)
        # writer.add_scalar('val_precision_score', precisionscore, global_step=epoch * val_epoch + batch_index)
        # writer.add_scalar('val_acc_score', accscore, global_step=epoch * val_epoch + batch_index)
        # writer.add_scalar('val_recall_score', recallscore, global_step=epoch * val_epoch + batch_index)
        ################################

        f1_value.update(f1score)
        precision_value.update(precisionscore)
        acc_value.update(accscore)
        recall_value.update(recallscore)
        #
        # f1_value_band.update(f1score_band)
        # precision_value_band.update(precisionscore_band)
        # acc_value_band.update(accscore_band)
        # recall_value_band.update(recallscore_band)
        #
        # f1_value_cls.update(f1score_cls)
        # precision_value_cls.update(precisionscore_cls)
        # acc_value_cls.update(accscore_cls)
        # recall_value_cls.update(recallscore_cls)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, val_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'area_f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
                   'area_precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value) + \
                   'area_acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) + \
                   'area_recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)
                   #  'band_f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_band) + \
                   #  'band_precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value_band) + \
                   #  'band_acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_band) + \
                   #  'band_recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_band) + \
                   # 'cls_f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_cls) + \
                   # 'cls_precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value_cls) + \
                   # 'cls_acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_cls) + \
                   # 'cls_recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_cls)

            print(info)

        if batch_index >= val_epoch:
            break

    return {'f1_avg': f1_value.avg,
            'precision_avg': precision_value.avg,
            'accuracy_avg': acc_value.avg,
            'recall_avg': recall_value.avg,

            'f1_band_avg': 0,
            'precision_band_avg': 0,
            'accuracy_band_avg': 0,
            'recall_band_avg': 0,

            'f1_cls_avg': 0,
            'precision_cls_avg': 0,
            'accuracy_cls_avg': 0,
            'recall_cls_avg': 0,
           }





if __name__ == '__main__':
    main()