"""
@created by HDG
用来计算各个数据集的指标
输入是模型 output 的文件夹目录 和 数据集原图 和gt
输出是四个指标的计算结果
指标结果数据输出为csv文本
"""
import os,sys,time,warnings,traceback
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

class Evaluation:
    def __init__(self,result_dir,src_dir,gt_dir):
        pass

    def match_rule(self):
        pass




if __name__ == '__main__':
    data_dict = {
        # 公开数据集
        'columbia':[

        ],
        'coverage':[

        ],
        'casia':[

        ],
        'realistic':[

        ],
        # 合成数据集
        'coco':[

        ],

    }

    # path check

    result_dir = data_dict['columbia'][0]
    src_dir = data_dict['columbia'][1]
    gt_dir = data_dict['columbia'][2]
    evaler = Evaluation(result_dir=result_dir, src_dir=src_dir,gt_dir=gt_dir)

    # 批量测试
if __name__ == '__main__':
    pass