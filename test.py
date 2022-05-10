"""
created by HDG
time 2022-1-8
"""
import traceback
from model.unet_model_aspp_attention import UNet as Net
import os, sys
import numpy as np
from PIL import Image
# import shutil
# import argparse
# import time
# import datetime
import torch

import cv2 as cv
# from functions import sigmoid_cross_entropy_loss, cross_entropy_loss,l1_loss,wce_huber_loss
# from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
# from utils import to_none_class_map
device = torch.device("cpu")

def read_test_data(output_path):
    try:
        image_name = os.listdir(test_data_path)
        length = len(image_name)
        for index,name in enumerate(image_name):
            print(index,'/',length)
            # name = 'Default_78_398895_clock.png'
            # gt_name = name.replace('Default','Gt')
            # gt_name = gt_name.replace('png','bmp')
            # gt_name = gt_name.replace('jpg','bmp')
            # gt = Image.open(os.path.join(test_data_path,gt_name))
            # gt = np.array(gt)
            # gt = to_none_class_map(gt)
            # plt.figure('gt')
            # plt.imshow(gt)
            # plt.show()

            image_path = os.path.join(test_data_path,name)
            # gt_path = Helper().find_gt(image_path)

            # gt = Image.open(gt_path)
            # gt = np.array(gt)
            # plt.figure('gt')
            # plt.imshow(gt)
            # plt.show()
            # gt = np.where((gt == 255) | (gt == 100), 1, 0)
            # plt.figure('gt2')
            # plt.imshow(gt)
            # plt.show()
            # gt_ = np.array(gt,dtype='float32')
            try:
                img = Image.open(image_path)
            except Exception as e:
                print(e)
                continue
            # resize 的方式
            if img.size != (320, 320):
                # try:
                #     img = Image.fromarray(img)
                # except Exception as e:
                #     print(e)
                #     continue
                img = img.resize((320,320))
                img = np.array(img,dtype='uint8')

            img = np.array(img,dtype='float32')

            img[:,:,0] =img[:,:,0]-0.47*255
            img[:, :, 1] = img[:, :, 1] - 0.43*255
            img[:, :, 2] = img[:, :, 2] - 0.39*255
            img[:, :, 0] /= 0.27 * 255
            img[:, :, 1] /= 0.26 * 255
            img[:, :, 2] /= 0.27 * 255
            img = np.transpose(img,(2,0,1))
            img = img[np.newaxis,:,:,:]
            img = torch.from_numpy(img)
            img = img.cpu()
            # print(img.shape)
            output = model(img)
            output = output[0]

            output = np.array(output.cpu().detach().numpy(), dtype='float32')
            output = output.squeeze(0)



            # output = np.transpose(output,(1,2,0))
            # output_ = output.squeeze(2)

            # 在这里计算一下loss
            # output_ = torch.from_numpy(output_)
            # gt_ = torch.from_numpy(gt_)
            # loss = wce_huber_loss(output_,gt_)
            # print(loss)
            # plt.figure('prediction')
            # plt.imshow(output_)
            # plt.show()
            output = np.array(output)*255
            output = np.asarray(output,dtype='uint8')
            # output_img = Image.fromarray(output)
            # output_img.save(output_path + 'output_'+name)
            cv.imwrite(os.path.join(output_path,'output_'+name),output)
    except Exception as e:
        traceback.print_exc()
        print(e)

# class Helper():
#     def __init__(self,test_src_dir = '/media/liu/File/11月数据准备/CASIA2.0_DATA_FOR_TRAIN/src',
#                  test_gt_dir ='/media/liu/File/11月数据准备/CASIA2.0_DATA_FOR_TRAIN/gt'):
#         self.test_src_dir = test_src_dir
#         self.test_gt_dir = test_gt_dir
#         pass
#     def find_gt(self, src_path):
#         """
#         using src name to find gt
#         using this function to validation loss
#         using this funciton when debug
#         :return: gt path
#         """
#         src_name = src_path.split('/')[-1]
#         # gt_name = src_name.replace('Default','Gt').replace('png','bmp').replace('jpg','bmp')
#         gt_name = src_name.split('.')[0] + '_gt.png'
#         gt_path = os.path.join(self.test_gt_dir,gt_name)
#
#         if os.path.exists(gt_path):
#             pass
#         else:
#             print(gt_path,'not exists')
#             traceback.print_exc()
#             sys.exit()
#         return gt_path

if __name__ == '__main__':
    try:

        test_data_path = '/home/liu/haoran/3月最新数据/public_dataset/columbia/src'
        output_path = '/home/liu/haoran/test_result/xr-2-columbia'
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(output_path)


        model_zoo = {
            'naive_unet':'/home/liu/haoran/HDG_ImgTamperDetection/save_model/0108_naive_unet_area_coco_sp_cm/0108_naive_unet_area_checkpoint-77-0.280948-f10.647072-precision0.576892-acc0.921198-recall0.751415.pth',
            'band_weight':'/home/liu/haoran/HDG_ImgTamperDetection/save_model/0221_srm_aspp_dilate_attention_unet_area_coco_sp_cm_template_edge_weight/0221_unet_attention_area_checkpoint-61-0.101309-f10.933732-precision0.944914-acc0.985886-recall0.924695.pth'

        }
        model_path = model_zoo['band_weight']
        checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
        model = Net().to(device)
        # model = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        read_test_data(output_path)
    except Exception as e:
        traceback.print_exc()
        print(e)