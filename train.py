import torch
import torch.optim as optim
import torch.utils.data.dataloader
import os, sys
sys.path.append('')
sys.path.append('../utils')
import argparse
import time, datetime
from functions import my_f1_score, my_acc_score, my_precision_score, cross_entropy_loss_edge_weight, wce_huber_loss, \
    wce_huber_loss_8, my_recall_score, cross_entropy_loss, wce_dice_huber_loss
from torch.nn import init
from datasets.dataloader import TamperDataset
from PIL import Image
import shutil
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import Logger, Averagvalue, weights_init, load_pretrained
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
# from model.unet_model import UNet as Net
from model.unet_model_aspp_attention import UNet as Net



"""
Created by HDG
time: 2022/01/6
description:
1. stage one training
"""

""""""""""""""""""""""""""""""
"          参数               "
""""""""""""""""""""""""""""""

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=24, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--model_save_dir', type=str, help='model_save_dir',
                    default='save_model/0221_srm_aspp_dilate_attention_unet_area_coco_sp_cm_template_edge_weight')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=20, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=150, type=int, metavar='N',
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
parser.add_argument('--resume', default='/home/liu/haoran/HDG_ImgTamperDetection/save_model/0218_srm_aspp_dilate_attention_unet_area_coco_sp_cm_template/0218_unet_attention_area_checkpoint-99-0.052342-f10.880099-precision0.862926-acc0.973697-recall0.902359.pth', type=str, metavar='PATH',
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

# tensorboard 使用
writer = SummaryWriter('../runs/' + '0221_tensorboard')
output_name_file_name = '0221_unet_attention_area_checkpoint-%d-%f-f1%f-precision%f-acc%f-recall%f.pth'
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
                  'cod10k': False,
                  'casia': False,
                  'coverage': False,
                  'columb': False,
                  'negative_coco': False,
                  'negative_casia': False,
                  'texture_sp': False,
                  'texture_cm': False,
                  }
    using_data_test = {'my_sp': False,
                       'my_cm': False,
                       'template_casia_casia': False,
                       'template_coco_casia': False,
                       'cod10k': False,
                       'casia': False,
                       'coverage': True,
                       'columb': False,
                       'negative_coco': False,
                       'negative_casia': False,
                       }
    # 2 define 3 types
    trainData = TamperDataset(using_data=using_data, train_val_test_mode='train')
    valData = TamperDataset( using_data=using_data, train_val_test_mode='val')
    testData = TamperDataset( using_data=using_data_test, train_val_test_mode='test')

    # 3 specific dataloader
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=args.batch_size, num_workers=3, shuffle=True,
                                                  pin_memory=True)
    valDataLoader = torch.utils.data.DataLoader(valData, batch_size=args.batch_size, num_workers=3)

    testDataLoader = torch.utils.data.DataLoader(testData, batch_size=args.batch_size, num_workers=1)
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
        # test_avg = test(model=model, dataParser=testDataLoader, epoch=epoch)

        """"""""""""""""""""""""""""""
        "          写入图            "
        """"""""""""""""""""""""""""""
        try:
            writer.add_scalar('train_avg_loss_per_epoch', train_avg['loss_avg'], global_step=epoch)

            writer.add_scalar('val_avg_loss_per_epoch', val_avg['loss_avg'], global_step=epoch)
            writer.add_scalar('val_avg_f1_per_epoch', val_avg['f1_avg'], global_step=epoch)
            writer.add_scalar('val_avg_precision_per_epoch', val_avg['precision_avg'], global_step=epoch)
            writer.add_scalar('val_avg_acc_per_epoch', val_avg['accuracy_avg'], global_step=epoch)
            writer.add_scalar('val_avg_recall_per_epoch', val_avg['recall_avg'], global_step=epoch)
            #
            # writer.add_scalar('test_avg_loss_per_epoch', test_avg['loss_avg'], global_step=epoch)
            # writer.add_scalar('test_avg_f1_per_epoch', test_avg['f1_avg'], global_step=epoch)
            # writer.add_scalar('test_avg_precision_per_epoch', test_avg['precision_avg'], global_step=epoch)
            # writer.add_scalar('test_avg_acc_per_epoch', test_avg['accuracy_avg'], global_step=epoch)
            # writer.add_scalar('test_avg_recall_per_epoch', test_avg['recall_avg'], global_step=epoch)

            writer.add_scalar('lr_per_epoch', scheduler.get_lr(), global_step=epoch)
        except Exception as e:
            print(e)

        """"""""""""""""""""""""""""""
        "          写入图            "
        """"""""""""""""""""""""""""""

        # 保存模型
        """
                    info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.steps_per_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
                   'precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value) + \
                   'acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) +\
                   'recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)
        """
        output_name = output_name_file_name % \
                      (epoch, val_avg['loss_avg'],
                       val_avg['f1_avg'],
                       val_avg['precision_avg'],
                       val_avg['accuracy_avg'],
                       val_avg['recall_avg'])


        if epoch % 1 == 0:
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

    # switch to train mode
    model.train()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        # 准备输入数据
        images = input_data['tamper_image'].cuda()
        labels = input_data['gt'].cuda()
        gt_band = input_data['gt_band'].unsqueeze(1)

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
        else:
            loss = torch.zeros(1)

        with torch.set_grad_enabled(True):
            images.requires_grad = True
            optimizer.zero_grad()
            # 网络输出
            outputs = model(images)


            """"""""""""""""""""""""""""""
            "         Loss 函数           "
            """"""""""""""""""""""""""""""


            # loss = wce_dice_huber_loss(outputs, labels)
            loss = cross_entropy_loss_edge_weight(outputs, labels,gt_band.cuda())
            loss.backward()
            optimizer.step()

        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        # f1score = my_f1_score(outputs[0], labels)
        # precisionscore = my_precision_score(outputs[0], labels)
        # accscore = my_acc_score(outputs[0], labels)
        # recallscore = my_recall_score(outputs[0], labels)
        #
        # writer.add_scalar('f1_score', f1score, global_step=epoch * train_epoch + batch_index)
        # writer.add_scalar('precision_score', precisionscore, global_step=epoch * train_epoch + batch_index)
        # writer.add_scalar('acc_score', accscore, global_step=epoch * train_epoch + batch_index)
        # writer.add_scalar('recall_score', recallscore, global_step=epoch * train_epoch + batch_index)
        ################################

        # f1_value.update(f1score)
        # precision_value.update(precisionscore)
        # acc_value.update(accscore)
        # recall_value.update(recallscore)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, train_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses)

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
    losses = Averagvalue()
    f1_value = Averagvalue()
    acc_value = Averagvalue()
    recall_value = Averagvalue()
    precision_value = Averagvalue()

    # switch to test mode
    model.eval()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)

        images = input_data['tamper_image']
        labels = input_data['gt']

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
        else:
            loss = torch.zeros(1)

        # 网络输出
        outputs = model(images)

        """"""""""""""""""""""""""""""
        "         Loss 函数           "
        """"""""""""""""""""""""""""""


        loss = wce_dice_huber_loss(outputs, labels)
        writer.add_scalar('val_fuse_loss_per_epoch', loss.item(),
                          global_step=epoch * val_epoch + batch_index)

        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score = my_f1_score(outputs, labels)
        precisionscore = my_precision_score(outputs, labels)
        accscore = my_acc_score(outputs, labels)
        recallscore = my_recall_score(outputs, labels)

        writer.add_scalar('val_f1_score', f1score, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_precision_score', precisionscore, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_acc_score', accscore, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_recall_score', recallscore, global_step=epoch * val_epoch + batch_index)
        ################################

        f1_value.update(f1score)
        precision_value.update(precisionscore)
        acc_value.update(accscore)
        recall_value.update(recallscore)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, val_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'vla_Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'val_f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
                   'val_precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value) + \
                   'val_acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) + \
                   'val_recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)

            print(info)

        if batch_index >= val_epoch:
            break

    return {'loss_avg': losses.avg,
            'f1_avg': f1_value.avg,
            'precision_avg': precision_value.avg,
            'accuracy_avg': acc_value.avg,
            'recall_avg': recall_value.avg}

#
# @torch.no_grad()
# def test(model, dataParser, epoch):
#     # 读取数据的迭代器
#     test_epoch = len(dataParser)
#
#     # 变量保存
#     batch_time = Averagvalue()
#     data_time = Averagvalue()
#     losses = Averagvalue()
#     f1_value = Averagvalue()
#     acc_value = Averagvalue()
#     recall_value = Averagvalue()
#     precision_value = Averagvalue()
#
#     # switch to test mode
#     model.eval()
#     end = time.time()
#
#     for batch_index, input_data in enumerate(dataParser):
#         # 读取数据的时间
#         data_time.update(time.time() - end)
#
#         images = input_data['tamper_image']
#         labels = input_data['gt_band']
#
#         if torch.cuda.is_available():
#             images = images.cuda()
#             labels = labels.cuda()
#
#         if torch.cuda.is_available():
#             loss = torch.zeros(1).cuda()
#             loss_8t = torch.zeros(()).cuda()
#         else:
#             loss = torch.zeros(1)
#             loss_8t = torch.zeros(())
#
#         # 网络输出
#         outputs = model(images)
#
#         """"""""""""""""""""""""""""""
#         "         Loss 函数           "
#         """"""""""""""""""""""""""""""
#
#         loss = wce_dice_huber_loss(outputs[0], labels)
#         writer.add_scalar('val_fuse_loss_per_epoch', loss.item(),
#                           global_step=epoch * test_epoch + batch_index)
#
#         # 将各种数据记录到专门的对象中
#         losses.update(loss.item())
#         map8_loss_value.update(loss_8t.item())
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         # 评价指标
#         f1score = my_f1_score(outputs[0], labels)
#         precisionscore = my_precision_score(outputs[0], labels)
#         accscore = my_acc_score(outputs[0], labels)
#         recallscore = my_recall_score(outputs[0], labels)
#
#         writer.add_scalar('test_f1_score', f1score, global_step=epoch * test_epoch + batch_index)
#         writer.add_scalar('test_precision_score', precisionscore, global_step=epoch * test_epoch + batch_index)
#         writer.add_scalar('test_acc_score', accscore, global_step=epoch * test_epoch + batch_index)
#         writer.add_scalar('test_recall_score', recallscore, global_step=epoch * test_epoch + batch_index)
#         writer.add_images('test_image_batch:%d' % (batch_index), outputs[0], global_step=epoch)
#         ################################
#
#         f1_value.update(f1score)
#         precision_value.update(precisionscore)
#         acc_value.update(accscore)
#         recall_value.update(recallscore)
#
#         if batch_index % args.print_freq == 0:
#             info = 'TEST_Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, test_epoch) + \
#                    'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
#                    'test_Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
#                    'test_f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
#                    'test_precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(
#                        precision=precision_value) + \
#                    'test_acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) + \
#                    'test_recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)
#
#             print(info)
#
#         if batch_index >= test_epoch:
#             break
#
#     return {'loss_avg': losses.avg,
#             'f1_avg': f1_value.avg,
#             'precision_avg': precision_value.avg,
#             'accuracy_avg': acc_value.avg,
#             'recall_avg': recall_value.avg}
#

if __name__ == '__main__':
    main()