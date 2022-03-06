import numpy as np
from PIL import Image

gt = np.zeros((320,320,3))
gt = np.array(gt,dtype='uint8')
gt = Image.fromarray(gt)
gt.save('negative_gt.bmp')

