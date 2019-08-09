#imports
import json
import time
import pickle
import scipy.misc
#import skimage.io
import caffe

import numpy as np
import os.path as osp

from PIL import Image
import skimage
from skimage.morphology import disk
import matplotlib.image as  mpimg
import random

def forward2backwardMap(forward_flow):
    backward_map = np.zeros((200,200))
    for x in range(0,200):
        for y in range(0,200):
            if((forward_flow[0,x,y]!=0)|(forward_flow[1:,x,y]!=0)):
                uu = x+forward_flow[1,x,y]
                vv = y+forward_flow[0,x,y]
                uu = int(uu+0.5)
                vv = int(vv+0.5)
                
                if(uu<0):
                    continue
                if(vv<0):
                    continue
                if(uu>=200):
                    continue
                if(vv>=200):
                    continue
                        
                backward_map[uu,vv] = 1
                
    return backward_map

class MN_data_layer(caffe.Layer):
    
    def setup(self, bottom, top):
        
        self.top_names = ['srcImg', 'srcFlow', 'tform', 'tgtMask', 'srcMask']
        
        # analysis pamameters
        params = eval(self.param_str)
        
        check_params(params)

        self.batch_size = params['batch_size']
        
        self.batch_loader = BatchLoader(params, None)
        
        top[0].reshape(self.batch_size, 1, params['image_size'][0], params['image_size'][1])
        top[1].reshape(self.batch_size, params['num_tform'])
        top[2].reshape(self.batch_size, 1, params['image_size'][0], params['image_size'][1])
        top[3].reshape(self.batch_size, 1, params['image_size'][0], params['image_size'][1])
        top[4].reshape(self.batch_size, 2, params['image_size'][0], params['image_size'][1])
        
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            flow_mask, tgt_mask, v_tform, src_mask, gt_flow_F = \
            self.batch_loader.load_next_tuple()

            top[0].data[itt, ...] = flow_mask.astype(np.float)
            top[1].data[itt, ...] = v_tform
            top[2].data[itt, 0, ...] = tgt_mask.astype(np.float)
            top[3].data[itt, 0, ...] = src_mask.astype(np.float)
            top[4].data[itt, ...] = gt_flow_F.astype(np.float)
    def backward(self, top, propagate_down, bottom):
        pass

    
# ===============================
# ===== some tool functions =====
# ===============================

class BatchLoader(object):
    
    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.data_list_dir = params['data_list_dir']
        self.num_tform = np.int(params['num_tform'])
        self.image_size = params['image_size']
        self.data_num = np.int(params['data_num'])
        self._cur = 0

        # make index list
        self.IndexList = [0]
        for i_ind in range(0,self.data_num):
            self.IndexList.append(i_ind)
            #if (i_ind%18==4):
            #    self.IndexList.append(i_ind)

        # shuffle the index list
        random.shuffle(self.IndexList)

        # read All Tuple List (26 as a batch)
        f_all_data_list = open(self.data_list_dir)
        self.all_data_list = f_all_data_list.read().splitlines()

        print ("BatchLoader initialized with {} images".format(np.size(self.IndexList)))
    
    def load_next_tuple(self):
        if self._cur == (np.size(self.IndexList)-1):
            random.shuffle(self.IndexList)
            self._cur = 0

        # load data list
        data_list = self.all_data_list[(self.IndexList[self._cur])*26:(self.IndexList[self._cur])*26+26]

        # 4 load target mask image
#        tgt_mask = np.asarray(Image.open(data_list[3]))
#        print(tgt_mask[100,100])
        tgt_mask=(mpimg.imread(data_list[3])*255).astype(int)
#        print(tgt_mask[100,100])
#        print(tgt_mask.shape)
        tgt_mask = tgt_mask[:,:,3]
        tgt_mask.setflags(write=1)
        
        low_values_indices = tgt_mask < 128
        tgt_mask[low_values_indices] = 0
        high_values_indices = tgt_mask >= 128
        tgt_mask[high_values_indices] = 255
        
        # 5 load one-hot vector
        v_tform = np.zeros(self.num_tform)
        v_tform[np.int(data_list[4])] = 1
        

        # 12 load src mask
        src_mask = np.asarray(Image.open(data_list[11]))
        
        src_mask = src_mask[:,:,3]
        src_mask.setflags(write=1)
        
        low_values_indices = src_mask < 128
        src_mask[low_values_indices] = 0
        high_values_indices = src_mask >= 128
        src_mask[high_values_indices] = 255


        # load ground truth flow (forward)
        gt_flow_F = np.zeros((2,200,200))
        
#        flow_u = np.fromfile(open(data_list[22]), dtype = np.double)
        flow_u=np.load(data_list[22])
        flow_u.resize(200,200)
#        print(np.max(flow_u))
#        gt_flow_F[0,:,:] = flow_u.transpose()
        gt_flow_F[0,:,:] = flow_u

#        flow_v = np.fromfile(open(data_list[23]), dtype = np.double)
        flow_v=np.load(data_list[23])
        flow_v.resize(200,200)
#        gt_flow_F[1,:,:] = flow_v.transpose()
        gt_flow_F[1,:,:] = flow_v
        
        gt_flow_F[0][src_mask==0] = 0
        gt_flow_F[1][src_mask==0] = 0

        # turn forward flow to backward flow mask
        
        tform_num = int(data_list[4])
        index_num = int(data_list[1])
#        print(tform_num,index_num)
#        print(data_list[22])
#        index_num = int(data_list[0][49:(len(data_list[0])-8)])
#        flow_mask = np.asarray(Image.open('/media/hao/mySpaceA/VSPV/data_for_training/pred_flow_mask/{}_{}.png'.format(index_num, tform_num)))
        flow_mask=np.load('/home/lzy/data/PVHM_0-999/transpose_forward_mask/{}_{}.npy'.format(index_num, tform_num))
        # now finished! push the cursor
#        print(np.max(flow_mask))
        self._cur = self._cur + 1
#        print(data_list[4],data_list[1],data_list[22],data_list[3],data_list[11])
        return flow_mask, tgt_mask, v_tform, src_mask, gt_flow_F


# borrow from python layers from Caffe Pascal Tutorial
def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['batch_size', 'data_list_dir', 'num_tform', 'image_size', 'data_num']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
