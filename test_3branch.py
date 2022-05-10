"""
created by HDG
time 2022-1-8
"""
import traceback
import warnings

from model.unet_model_aspp_attention_three_branch import UNet as Net
import os, sys
import numpy as np
from PIL import Image
import shutil
import argparse
import time
import datetime
import torch

import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
warnings.filterwarnings('ignore')
device = torch.device("cpu")

def read_test_data(output_path,isshow=True):
    try:
        image_name = os.listdir(test_data_path)
        length = len(image_name)
        for index,name in enumerate(image_name):
            print(index,'/',length)
            image_path = os.path.join(test_data_path,name)
            try:
                img = Image.open(image_path)
                img = img.convert("RGB")
                im0 = img.copy()
                if len(img.split()) == 4:
                    img = img.convert("RGB")

            except Exception as e:
                print(e)
                continue
            # # resize 的方式
            # if img.size != (320, 320):
            #     img = img.resize((320,320))
            #     img = np.array(img,dtype='uint8')

            img = np.array(img,dtype='float32')
            # (0.47, 0.43, 0.39), (0.27, 0.26, 0.27)
            # R_MEAN = img[:,:,0].mean()
            # G_MEAN = img[:,:,1].mean()
            # B_MEAN = img[:,:,2].mean()
            # img[:,:,0] =img[:,:,0]-R_MEAN
            # img[:, :, 1] = img[:, :, 1] - G_MEAN
            # img[:, :, 2] = img[:, :, 2] - B_MEAN
            # img[:, :, 0] /= 255
            # img[:, :, 1] /= 255
            # img[:, :, 2] /= 255


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


            out_area , out_band , out_cls = model(img)
            # # resize 的方式
            # if img.size != (320, 320):
            #     img = img.resize((320,320))
            #     img = np.array(img,dtype='uint8')

            out_area = np.array(out_area.cpu().detach().numpy(), dtype='float32').squeeze(0).squeeze(0)

            out_band = np.array(out_band.cpu().detach().numpy(), dtype='float32').squeeze(0).squeeze(0)


            out_area = np.array(out_area)*255
            out_band = np.array(out_band)*255
            out_area = np.asarray(out_area,dtype='uint8')
            out_band = np.asarray(out_band, dtype='uint8')

            #
            # if isshow:
            #
            #     plt.subplot(131)
            #     plt.imshow(im0)
            #     plt.subplot(132)
            #     plt.imshow(out_area)
            #     plt.subplot(133)
            #     plt.imshow(out_band)
            #     plt.title('The cls score is:{:.2f}'.format(100 * float(out_cls.cpu().detach().numpy())))
            #
            #     plt.show()


            cv.imwrite(os.path.join(output_path, 'area_' + name), out_area)
            cv.imwrite(os.path.join(output_path,'band_'+name), out_band)

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
    test_data_path = {
        'columbia':'/home/liu/haoran/3月最新数据/public_dataset/columbia/src',
        'coverage':'/home/liu/haoran/3月最新数据/public_dataset/coverage/src',
        'casia':'/home/liu/haoran/3月最新数据/public_dataset/casia/src',
        'realistic':'/home/liu/haoran/3月最新数据/public_dataset/realistic/src',
        'wild':'/home/liu/haoran/3月最新数据/public_dataset/wild/images',
        'ps':'/home/liu/haoran/3月最新数据/public_dataset/ps-battle-30',
        'coco':'/home/liu/haoran/3月最新数据/coco_sp/test_src',
        'negative':'/home/liu/haoran/3月最新数据/negative',
    }


    try:
        test_data_path = test_data_path['wild']
        output_path = '/home/liu/haoran/test_result/xr-4-wild'
        model_path = '/home/liu/haoran/HDG_ImgTamperDetection/save_model/0304_3branch/0303_3branch-99-0.070301-[f10.802046-precision0.702137-acc0.976859-recall0.945557]-[f10.531083-precision0.369580-acc0.976056-recall0.961372]-[f10.531083-precision0.369580-acc0.976056-recall0.961372].pth'
        # mkdir
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(output_path)
        checkpoint = torch.load(model_path,map_location=torch.device('cpu'))

        model = Net().to(device)
        # model = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        read_test_data(output_path,isshow=True)
    except Exception as e:
        traceback.print_exc()
        print(e)