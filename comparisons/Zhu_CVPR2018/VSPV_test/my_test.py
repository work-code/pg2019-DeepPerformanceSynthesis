#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:59:11 2019

@author: lzy
"""




# coding: utf-8

# In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

import skimage
from skimage.morphology import disk
import sys
sys.path.append('./utility/')



import matplotlib.image as  mpimg
import numba as nb
#import pycaffe
sys.path.append('./caffe_VSPV/python/')
import caffe

caffe.set_mode_gpu()

sys.path.append("./model/python_layers/")
DN = caffe.Net('./model/DepthNet_deploy.prototxt','./snapshot/DN/iter_iter_200000.caffemodel', caffe.TEST)
FN = caffe.Net('./model/new_FlowNet_deploy .prototxt','./snapshot/FN/iter_iter_170000.caffemodel', caffe.TEST)
MN = caffe.Net('./model/MaskNet_deploy.prototxt','./snapshot/MN/iter_iter_100000.caffemodel', caffe.TEST)


@nb.jit
def DepthMap2XyzImage(depth_map,K,Rt):
    
    P = np.dot(K,Rt)
#    C = np.dot(-(Rt[0:3,0:3]).transpose(), Rt[0:3,3])
    P_inv = np.linalg.pinv(P)
    C9 = np.dot(-(Rt[0:3,0:3]).transpose(), Rt[0:3,3])
    
    uv3 = np.array([100.5,100.5,1])
    xyz3 = np.dot(P_inv,uv3)
    xyz3 = xyz3 / xyz3[3]
    dir_C = xyz3[0:3] - C9
    dir_C = dir_C/np.linalg.norm(dir_C)    
    
    new_xyz_image = np.zeros((3,200,200))
    for xx in range(0,200):
        for yy in range(0,200):
            if depth_map[xx,yy] == 0:
                continue

            uv1 = np.array([yy+1,xx+1,1])
            xyz1 = np.dot(P_inv,uv1)
            xyz1 = xyz1 / xyz1[3]
            dir_ = xyz1[0:3] - C9
            dir_ = dir_ / np.linalg.norm(dir_)


            cos = np.dot(dir_,dir_C)

            depth = depth_map[xx,yy]/cos

            new_xyz = dir_*depth + C9
            new_xyz_image[:,xx,yy] = new_xyz
            
    return new_xyz_image

@nb.jit
def xyzImage2ForwardFlow(gt_xyz, K, Rt):
    P = np.dot(K,Rt)
    forward_flow = np.zeros((2,200,200))
    for x in range(0,200):
        for y in range(0,200):
            if np.sum(gt_xyz[:,x,y])==0:
                continue
            #if np.sum(gt_flow_F[:,x,y])==0:
            #    continue
            point = gt_xyz[:,x,y]
            
            point_1 = np.zeros(4)
            point_1[0:3] = point
            point_1[3] = 1
            
            UV = np.dot(P,point_1.transpose())
            UV = UV/UV[2]
            forward_flow[:,x,y] = UV[0:2] -1 -np.array([y,x])
            
    return forward_flow

def getKRtCArray():
    ex_mat = np.loadtxt('/home/lzy/data/PVHM_Tools/flow_renderer/extrinsic.txt')
    in_mat = np.loadtxt('/home/lzy/data/PVHM_Tools/flow_renderer/intrinsic.txt')
    
    K = in_mat[0:3,:]

    Rt_arr = np.zeros((36,3,4))
    C_arr = np.zeros((19,3))
    for i in range(0,19):
        Rt_arr[i,:] = ex_mat[i*3:i*3+3,:]
        R = ex_mat[i*3:i*3+3,:3]
        t = ex_mat[i*3:i*3+3,3]
        C_arr[i,:] = -np.dot(R.transpose(),t)
    for i in range(19,36):
        Rt_arr[i]=Rt_arr[i-18]
        Rt_arr[i,0,0:3]=-Rt_arr[i,0,0:3]
        Rt_arr[i,2,0:3]=-Rt_arr[i,2,0:3]
        
    return K,Rt_arr,C_arr

K, Rt_arr, C_arr = getKRtCArray()


from skimage.morphology import disk

@nb.jit
def forward2backwardMap(forward_flow):
    backward_map = np.zeros((200,200))
    for x in range(0,200):
        for y in range(0,200):
            if((forward_flow[0,x,y]!=0)|(forward_flow[1,x,y]!=0)):
                uu = x+forward_flow[1,x,y]
                vv = y+forward_flow[0,x,y]
                uu = int(uu+0.5)
                vv = int(vv+0.5)
                
                if uu<0 or vv<0 or uu>=200 or vv>=200:
                    continue
                        
                backward_map[uu,vv] = 1
                
    return backward_map


def getKRtC(num):
    ex_mat = np.loadtxt('/home/lzy/data/PVHM_Tools/flow_renderer/extrinsic.txt')
    in_mat = np.loadtxt('/home/lzy/data/PVHM_Tools/flow_renderer/intrinsic.txt')
    
    K = in_mat[num*3:num*3+3,:]
    Rt = ex_mat[num*3:num*3+3,:]
    R = ex_mat[num*3:num*3+3,:3]
    t = ex_mat[num*3:num*3+3,3]
    C = -np.dot(R.transpose(),t)
    return K,Rt,C


# In[124]:
import time
#mid为中间图片的序号    
mid=27
mm=mid/9
print(mm)
for mm in range(1,4):
    mid=9*mm
    for i in range(9200,9400):
        for j in range(0,18):
                 
            
            time1=time.time()
            src_mask=(mpimg.imread('/home/lzy/data/PVHM_0-9999/rendered_images/{}_{}_200x200.png'.format(i,mid))*255).astype(int)  
     
            src_mask = src_mask[:,:,3]
            src_mask.setflags(write=1)
            low_values_indices = src_mask < 128
            src_mask[low_values_indices] = 0
            high_values_indices = src_mask >= 128
            src_mask[high_values_indices] = 1
            
            
      
            src_img=(mpimg.imread('/home/lzy/data/PVHM_0-9999/rendered_images/{}_{}_200x200.png'.format(i,mid))*255)
            src_img = src_img[:,:,:3]
            src_img=np.rollaxis(src_img,2,0)
            
            print(1,time.time()-time1)
            time1=time.time()
            
            #
            DN.blobs['srcImg'].data[0] = src_img
            DN.forward()
            predDepth = DN.blobs['final'].data[0,0]
            predDepth = predDepth * src_mask
            #
            ## depth 2 forward flow
            print(2,time.time()-time1)
            
            time1=time.time()
            
            (K,Rt,C)=getKRtC(0)
            xyz_image= DepthMap2XyzImage(predDepth, K, Rt_arr[mid])
            forward_flow = xyzImage2ForwardFlow(xyz_image, K, Rt_arr[j+mid-9])
    #        print(np.max(forward_flow))
            forward_flow[0][src_mask==0] = 0
            forward_flow[1][src_mask==0] = 0
            
            
            print(3,time.time()-time1)
            time1=time.time()
           
            
            FN.blobs['forward_flow'].data[0] = forward_flow
            FN.blobs['gt_xyz'].data[0] = xyz_image
    #        FN.blobs['srcImg'].data[0] = src_img
            FN.blobs['tform'].data[0] = j
    #        FN.blobs['backward_flow'].data[0] = gt_flow_B
    
    #        FN.blobs['tgtImg'].data[0] = tgt_img
            print(4.1,time.time()-time1)
            time1=time.time()
    
    #        
            FN.forward()
            print(4.2,time.time()-time1)
            time1=time.time()
            remaped_forward_flow=np.array(FN.blobs['remaped_forward_flow'].data[0])
            backward_flow=np.array(FN.blobs['coords'].data)
            
            print(4.3,time.time()-time1)
            time1=time.time()
            
            np.save('./backward_flow_200/{}_{}_{}_200'.format(mm,i,j),backward_flow)

            
            
            
            
            flow_mask = forward2backwardMap(forward_flow)
            flow_mask = skimage.morphology.closing(flow_mask,disk(2)) * 255
            flow_mask = flow_mask
            #
            forward_flow[0][src_mask==0] = 0
            forward_flow[1][src_mask==0] = 0
         
            print(5,time.time()-time1)
            time1=time.time()
        
            MN.blobs['srcMask'].data[0,0] = src_mask
            MN.blobs['srcFlow'].data[0] = forward_flow
            MN.blobs['tform'].data[0] = j        
            MN.forward()       
            predMask = MN.blobs['predMask'].data[0,0]
            
    
            print(6,time.time()-time1)
            time1=time.time()
            
            finalMask = predMask.astype(np.float) + flow_mask.astype(np.float)
            finalMask[finalMask>0] = 255
            np.save('./finalmask_200/{}_{}_{}'.format(mm,i,j),finalMask)
            
            print(7,time.time()-time1)
    	print('./finalmask_200/{}_{}_{}'.format(mm,i,j))


