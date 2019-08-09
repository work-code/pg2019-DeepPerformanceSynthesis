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

sum1=0
sum2=0
sum3=0
for i in range(1900,1950):
    for j in range(18):
        if j!=9:
           a=mpimg.imread('/home/lzy/data/PVHM_0-9999/rendered_images/{}_{}_200x200.png'.format(i,j))[:,:,0:3]
           b=mpimg.imread('./final_image/{}/{}_{}_1.png'.format(i,i,j))[:,:,0:3]
           c=mpimg.imread('./final_image/{}/{}_{}_2.png'.format(i,i,j))[:,:,0:3]
           d=mpimg.imread('./final_image/{}/{}_{}_3.png'.format(i,i,j))[:,:,0:3]
           sum1=sum1+np.sum(np.abs(a-b).squeeze())
           sum2=sum2+np.sum(np.abs(a-c).squeeze())
           sum3=sum3+np.sum(np.abs(a-d).squeeze())
print sum1
print sum2
print sum3           