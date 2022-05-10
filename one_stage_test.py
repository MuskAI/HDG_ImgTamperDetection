
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：forgery-edge-detection 
@File    ：one_stage_test.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/3/30 8:41 PM 
'''
import os, sys, random

import numpy as np
from tqdm import tqdm
from prepare_datasets import Datasets
from PIL import Image
import torchvision
import cv2 as cv
import traceback
from importlib import import_module
import torch
from pprint import pprint

"""
选择模型
"""
from model.final_model import UNet as Net

class OneStageInfer:
    def __init__(self, model):
        """
        只负责输入
        @param model_name:
        @param src_data_dir:
        @param output_dir:
        """
        self.model = model
        self.error_log = []

    def read_test_data(self, src_data_dir, output_dir, name_list=None):
        test_data_path = src_data_dir
        image_name = os.listdir(test_data_path) if name_list is None else name_list
        try:
            for index, name in enumerate(tqdm(image_name)):
                img = Image.open(os.path.join(test_data_path, name)).convert('RGB')
                img = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)
                if device == 'cuda':

                    img = img[np.newaxis, :, :, :].cuda()
                else:
                    img = img[np.newaxis, :, :, :].cpu()
                output = self.model(img)
                output = np.array(output[0].cpu().detach().numpy(), dtype='float32')
                output = output.squeeze(0)
                output = np.transpose(output, (1, 2, 0))
                output_ = output.squeeze(2)
                output = np.array(output_) * 255
                output = np.asarray(output, dtype='uint8')
                cv.imwrite(os.path.join(output_dir, '{}.png'.format(name.split('.')[0])), output)
        except Exception as e:
            traceback.print_exc()
            print(e)


class TestDatasets(Datasets):
    def __init__(self):
        self.using_data = {'columbia': True,
                           'coverage': True,
                           'coverage': True,
                           'casia': True,
                           'ps-battle': False,
                           'in-the-wild': False,
                           }

        self.datasets_path = {
            'root': '/home/liu/haoran/3月最新数据/public_dataset',
            'columbia': None,
            'coverage': None,
            'casia': None,
            'my-protocol': None
        }
        super(TestDatasets, self).__init__(model_name='HDG-final-0418-casiafinetune', using_data=self.using_data,
                                           datasets_path=self.datasets_path)
        self.datasets_dict = self.get_datasets_dict()
        # pprint(self.datasets_dict)
        self.one_stage_infer = OneStageInfer(model=self.get_model())
        self.infer_all_dataset()

    def infer_all_dataset(self):
        for idx, item in enumerate(tqdm(self.datasets_dict)):
            # 遍历每张图片
            self.one_stage_infer.read_test_data(src_data_dir=self.datasets_dict[item]['path'],
                                                output_dir=self.datasets_dict[item]['save_path'],
                                                name_list=self.datasets_dict[item]['names'])

    def get_model(self):
        model_path = self.model_zoo()['HDG-final-0418']
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model = Net().to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def model_zoo(self):
        model_dict = {
            'HDG-final': '/home/liu/haoran/HDG_ImgTamperDetection/save_model/0304_3branch/0303_3branch-149-0.070555-[f10.799064-precision0.696681-acc0.976385-recall0.947282]-[f10.524830-precision0.363389-acc0.975457-recall0.961997]-[f10.524830-precision0.363389-acc0.975457-recall0.961997].pth',
            'HDG-final-finetune':'/home/liu/haoran/HDG_ImgTamperDetection/save_model/0331_3branch_area_band/0303_3branch-17-0.144680-[f10.818465-precision0.736004-acc0.971634-recall0.925340]-[f10.526044-precision0.365569-acc0.965829-recall0.943173]-[f10.526044-precision0.365569-acc0.965829-recall0.943173].pth',
            'HDG-aspp-attention':'/home/liu/haoran/HDG_ImgTamperDetection/save_model/aspp_atention_model/0404-90-0.002716-[f10.880087-precision0.844396-acc0.982605-recall0.922103]-[f10.000000-precision0.000000-acc0.000000-recall0.000000]-[f10.000000-precision0.000000-acc0.000000-recall0.000000].pth',
            'HDG-final-0418':'/home/liu/haoran/HDG_ImgTamperDetection/save_model/final_model_finetune/0418-22-0.061576-[f10.732017-precision0.668039-acc0.953510-recall0.861029]-[f10.418137-precision0.279323-acc0.943100-recall0.932287]-[f10.418137-precision0.279323-acc0.943100-recall0.932287].pth',
            }

        return model_dict

if __name__ == '__main__':
    device = 'cpu'
    tester = TestDatasets()