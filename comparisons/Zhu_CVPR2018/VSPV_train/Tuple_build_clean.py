# new version: single tuple file
#coding:utf-8

import scipy.misc
import numpy as np
from PIL import Image
import random

count = 0

IndexList = []
for i_ind in range(0,30000):
    IndexList.append(i_ind)

random.shuffle(IndexList)
print('~/home/VSPV/Tuple_Standard_26_150000_train.txt')
f_tuple_train = open('/home/lzy/data/Tuple_Standard_26_150000_train.txt','w')
for num in range(30000):
    i = IndexList[num]/18
    j = IndexList[num]%18
    # index 0 Source Image
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/rendered_images/{}_9_200x200.png\n'.format(i))
    # index 1 X
    f_tuple_train.write('{}\n'.format(i))
    # index 2 target color image 
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/rendered_images/{}_{}_200x200.png\n'.format(i,j))
    # index 3 target mask
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/rendered_images/{}_{}_200x200.png\n'.format(i,j))
    # index 4 taget view index 
    f_tuple_train.write('{}\n'.format(j))
    # index 5 target backward flow - u
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/backward_flow/{}_{}_u.bin\n'.format(i,j))
    # index 6 target backward flow - v
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/backward_flow/{}_{}_v.bin\n'.format(i,j))
    # index 7 X (high resolution source image)
    f_tuple_train.write('\n')
    # index 8 X (high resolution target image)
    f_tuple_train.write('\n')
    # index 9 X (high resolution target mask)
    f_tuple_train.write('\n')
    # index 10 X
    f_tuple_train.write('\n')
    # index 11 source mask
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/rendered_images/{}_9_200x200.png\n'.format(i))
    # index 12 X
    f_tuple_train.write('\n')
    # index 13 X
    f_tuple_train.write('\n')
    # index 14 X
    f_tuple_train.write('\n')
    # index 15 X
    f_tuple_train.write('\n')
    # index 16 X
    f_tuple_train.write('\n')        
    # index 17 X
    f_tuple_train.write('\n')        
    # index 18 X
    f_tuple_train.write('\n')
    # index 19 X
    f_tuple_train.write('\n')
    # index 20 X
    f_tuple_train.write('\n')
    # index 21 depth image
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/depth/{}_9_d.bin\n'.format(i))
    # index 22 forward flow u
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/forward_flow/{}_{}_u.bin.npy\n'.format(i,j))
    # index 23 forward flow v
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/forward_flow/{}_{}_v.bin.npy\n'.format(i,j))
    # index 24 X
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/remapped_forward_flow/{}_{}_u.npy\n'.format(i,j))
    # index 25 X
    f_tuple_train.write('/home/lzy/data/PVHM_0-9999/remapped_forward_flow/{}_{}_v.npy\n'.format(i,j))


    if(count % 1000 == 0):
        print (count)

    count = count + 1
f_tuple_train.close()
