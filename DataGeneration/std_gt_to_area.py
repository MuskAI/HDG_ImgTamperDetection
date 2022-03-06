"""
created by HDG
time:2021-1-1
description:convert std gt to area gt
将std gt 转化为区域gt
"""

import argparse
import os,sys
from tqdm import tqdm
import cv2,PIL
import numpy as np
from PIL import Image
import traceback
parser = argparse.ArgumentParser()
parser.add_argument("--isfile" ,default=False, help="the path is a file")
parser.add_argument("--path" ,default="", help="gt path,one image path or image dir path")
parser.add_argument("--pixelvalue",type=list, default=[50,0,255,100],help="tamper area ,not tamper area, tamper edge, not tamper edge")

args = parser.parse_args()
print(args.isfile)


class Std2Area():
    """
    using this class to convert std gt to area gt
    """
    def __init__(self,isfile,path,pixelvalue):
        self.isfile = isfile
        self.path = path
        self.file_list = []
        self.file_path = ''
        self.pixelvalue = pixelvalue
        self.save_dir = ''

        # deal with file and dir issues
        if isfile:
            self.file_list = os.listdir(self.path)
        else:
            self.file_path = path

            self.save_dir = path.replace(path.split('/')[-1],path.split('/')[-1]+'_std2area')

        pass

    def __check(self):
        """
        i will implement check function if it is necessary
        :return:
        """
        pass
    def one_image_convert(self,gt_path):
        try:
            gt_img = Image.open(gt_path)

            # check img dims
            if len(gt_img.split()) !=1:
                gt_img = gt_img.split()[0]

            gt_img = np.array(gt_img,dtype='uint8')
            gt_img = np.where(gt_img==self.pixelvalue[2],self.pixelvalue[0])
            gt_img = np.where(gt_img==self.pixelvalue[3],self.pixelvalue[1])

            gt_img = Image.fromarray(gt_img)
            gt_name = gt_path.split('/')[-1]
            return gt_img,gt_name

        except Exception as e:
            traceback.print_exc(e)

    def all_image_convert(self):
        """
        convert all the image in dir,saving in bother dir default
        :param gt_list:
        :return:
        """
        gt_list = self.file_list
        for idx,gt_path in tqdm(enumerate(gt_list)):
            try:
                gt_img,gt_name = self.one_image_convert(gt_path)
                gt_img.save(os.path.join(self.save_dir,gt_name))
            except Exception as e:
                traceback.print_exc(e)


if __name__ == '__main__':
    converter = Std2Area(isfile=args.isfile,path=args.path,pixelvalue=args.pixelvalue)
    converter.all_image_convert()
