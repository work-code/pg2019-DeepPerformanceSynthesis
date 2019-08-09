# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 06:56:53 2019

@author: liuzhenye
"""

import project_functions
import numpy as np
import skimage
from skimage.morphology import disk
from PIL import Image
import numba as nb

import sys
sys.path.append('C:/Users/liuzhenye/Desktop/Project3/VSPV/model/python_layers/')
import forward_backward_map_layer
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

def forward_flow(path,number):
    for i in range(25,number):     
        depth_map=np.fromfile(open(path+'depth/{}_9_d.bin'.format(i)), dtype = np.double)
        print(depth_map.shape)
        depth_map.resize(200,200)
        (K,Rt,C)=project_functions.getKRtC(9)
        new_xyz_image=project_functions.DepthMap2XyzImage(depth_map,K,Rt)
        for j in range(19):
            (K,Rt,C)=project_functions.getKRtC(j)
            forward_flow=project_functions.xyzImage2ForwardFlow(new_xyz_image, K, Rt)            
            np.save(path+'forward_flow/{}_{}_u.bin'.format(i,j),forward_flow[0,:,:])
            np.save(path+'forward_flow/{}_{}_v.bin'.format(i,j),forward_flow[1,:,:])
        print(i)
@nb.jit        
def forward_mask(path,number):
    for i in range(25,number):
        for j in range(19):
            forward_flow=np.zeros([2,200,200])
            forward_flow[0,:,:]=np.load(path+'forward_flow/{}_{}_u.bin.npy'.format(i,j))
            forward_flow[1,:,:]=np.load(path+'forward_flow/{}_{}_v.bin.npy'.format(i,j))
            flow_mask = forward2backwardMap(forward_flow)
            flow_mask = skimage.morphology.closing(flow_mask,disk(2)) * 255
            flow_mask = flow_mask     
            np.save(path+'forward_mask/{}_{}'.format(i,j),flow_mask)
        print (i)
        
        
def reampped_forward_flow(path,number): 
    for i in range(number):             
        for j in range(19):
            bottom=np.zeros([2,200,200])
            bottom[0,:,:]=np.load(path+'forward_flow/{}_{}_u.bin.npy'.format(i,j))
            bottom[0,:,:]=np.load(path+'forward_flow/{}_{}_v.bin.npy'.format(i,j))  
            top=np.zeros([2,200,200])
            f=forward_backward_map_layer.forward_backward_map_layer()
            np.save(path+'reampped_forward_flow/{}_{}_u.bin'.format(i,j),forward_flow[0,:,:])
            np.save(path+'remapped_forward_flow/{}_{}_v.bin'.format(i,j),forward_flow[1,:,:])
        print(i)      
#forward_flow('H:/data/PVHM_8000-8999/',1000)
#forward_flow('H:/data/PVHM_0-999/',1000)
#forward_mask('H:/data/PVHM_0-999/',1000)
#flow_mask = np.loadtxt('H:/data/PVHM_0-999/forward_mask/{}_{}.png'.format(0,0))
#print(flow_mask)
reampped_forward_flow('H:/data/PVHM_0-999/',1000)