# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 19:54:05 2019

@author: lzy
"""
import numpy as np
from PIL import Image
import matplotlib.image as  mpimg
import PIL
import os
import shutil
import matplotlib.pyplot as plt

resolutuion=512
left=9400
right=9600

for i in range(left,right):
    print(i)
    for j in range(3):
        srcImg=mpimg.imread("/home/lzy/data/PVHM_0-9999/rendered_images/{}_{}_{}x{}.png".format(i,j*9+9,resolutuion,resolutuion))[:,:,0:3]
        plt.imsave("./final_image_{}/{}-{}/{}/{}.png".format(resolutuion,left,right,i,j+1),srcImg)  
        
# In[]
        







# In[]
for i in range(1900,1950):
    print i
    for j in range(18):
       shutil.copy("/home/lzy/data/PVHM_0-9999/rendered_images/{}_{}_200x200.png".format(i,j),"/home/lzy/result/final_image/{}".format(i))  
