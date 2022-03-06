"""
created by HDG
无篡改数据处理
"""
import os
import shutil

def rename(src_dir,target_dir):
    src_list = os.listdir(src_dir)
    for idx, item in enumerate(src_list):
        print(idx,'/',len(src_list))
        src_path = os.path.join(src_dir,item)
        target_path = os.path.join(target_dir, 'negative_hand_'+item)
        if os.path.isfile(src_path) and os.path.join(target_path):
            shutil.copy(src_path,target_path)
        else:
            print(target_path,'is not a file')


if __name__ == '__main__':
    src_dir = '/home/liu/haoran/coco/COCO-Hand-Big_Images'
    target_dir = '/home/liu/haoran/3月最新数据/negative'
    rename(src_dir,target_dir)