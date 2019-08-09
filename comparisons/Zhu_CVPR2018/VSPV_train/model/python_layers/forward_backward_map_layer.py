import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from PIL import Image


def setValue(tgt_flow,src_flow,xyz_map,ori_xy_mask,u,v,x,y,tform,C):
    
    if(tgt_flow[0,u,v]==0 and tgt_flow[1,u,v]==0):
        tgt_flow[0,u,v] = -src_flow[0,x,y]
        tgt_flow[1,u,v] = -src_flow[1,x,y]
        ori_xy_mask[0,u,v] = x
        ori_xy_mask[1,u,v] = y
    else:
        #print xyz_map[:,x,y].shape
        #print C.shape
        new_depth = np.linalg.norm(xyz_map[:,x,y] - C[int(tform)])
        ori_depth = np.linalg.norm(
            xyz_map[:,int(ori_xy_mask[0,u,v]),int(ori_xy_mask[1,u,v])] - C[int(tform)])
        if(new_depth<ori_depth):
            tgt_flow[0,u,v] = -src_flow[0,x,y]
            tgt_flow[1,u,v] = -src_flow[1,x,y]
            ori_xy_mask[0,u,v] = x
            ori_xy_mask[1,u,v] = y
        
class forward_backward_map_layer(caffe.Layer):
    
    def setup(self, bottom, top):
        self.bias_mat = np.zeros((2,200,200))
        for i in range(0,200):
            self.bias_mat[0,i,:] = np.array(range(0,200))
            self.bias_mat[1,:,i] = np.array(range(0,200))
        #pass
        
        ex_mat = np.loadtxt('/home/lzy/data/PVHM_Tools/flow_renderer/extrinsic.txt')
        self.C = np.zeros((18,3))
        for x in range(0,18):
            R = ex_mat[x*3:x*3+3,:3]
            t = ex_mat[x*3:x*3+3,3]
            self.C[x] = -np.dot(R.transpose(),t)
        #print self.C
    
    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].shape[0],2,200,200)

    def forward(self, bottom, top):
        for itt in range(bottom[0].shape[0]):
            ori_xy_mask = np.zeros((2,200,200))
            forward_flow = bottom[0].data[itt, ...]
            
            xyz_map = bottom[1].data[itt, ...]
            tform = int(bottom[2].data[itt])
            
            test_flow = np.zeros((2,200,200))
            for x in range(0,200):
                for y in range(0,200):
                    if((forward_flow[0,x,y]!=0)|(forward_flow[1:,x,y]!=0)):
                        uu = x+forward_flow[1,x,y]
                        vv = y+forward_flow[0,x,y]
                        #uu = int(uu+0.5)
                        #vv = int(vv+0.5)
                        uu = np.int(np.floor(uu))
                        vv = np.int(np.floor(vv))
 
                        if(uu<0):
                            continue
                        if(vv<0):
                            continue
                        if(uu>=199):
                            continue
                        if(vv>=199):
                            continue
                        
                        setValue(test_flow,forward_flow,xyz_map,ori_xy_mask,uu,vv,x,y,tform,self.C)
                        setValue(test_flow,forward_flow,xyz_map,ori_xy_mask,uu+1,vv,x,y,tform,self.C)
                        setValue(test_flow,forward_flow,xyz_map,ori_xy_mask,uu,vv+1,x,y,tform,self.C)
                        setValue(test_flow,forward_flow,xyz_map,ori_xy_mask,uu+1,vv+1,x,y,tform,self.C)

            top[0].data[itt, ...] = test_flow
            
    def backward(self, top, propagate_down, bottom):
        pass


# borrow from python layers from Caffe Pascal Tutorial
def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['batch_size', 'data_list_dir', 'num_tform', 'image_size', 'data_num']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
