#imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from PIL import Image

import random

class DN_data_layer(caffe.Layer):
    
    def setup(self, bottom, top):
        
        self.top_names = ['srcImg', 'Depth', 'tgtMask']
        
        # analysis pamameters
        params = eval(self.param_str)
        
        check_params(params)

        self.batch_size = params['batch_size']
        
        self.batch_loader = BatchLoader(params, None)
        
        top[0].reshape(self.batch_size, 3, params['image_size'][0], params['image_size'][1])
        top[1].reshape(self.batch_size, 1, params['image_size'][0], params['image_size'][1])
        top[2].reshape(self.batch_size, 1, params['image_size'][0], params['image_size'][1])
        
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            src_img, depth_map, tgt_mask = \
            self.batch_loader.load_next_tuple()

            top[0].data[itt, ...] = src_img.astype(np.float)
            top[1].data[itt, ...] = depth_map.astype(np.float)
            top[2].data[itt, 0, ...] = tgt_mask.astype(np.float)
            
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
        

        self.bias_mat = np.zeros((2,200,200))
        for i in range(0,200):
            self.bias_mat[0,i,:] = np.array(range(0,200))
            self.bias_mat[1,:,i] = np.array(range(0,200))        


        # make index list
        self.IndexList = [0]
        for i_ind in range(0,self.data_num):
            self.IndexList.append(i_ind)
            #if (i_ind%18==4):
            #    self.IndexList.append(i_ind)
        print  self.IndexList,111111111111111111111111111111111111111111111111111 
        # shuffle the index list
        random.shuffle(self.IndexList)

        # read All Tuple List (26 as a batch)
        f_all_data_list = open(self.data_list_dir)
        self.all_data_list = f_all_data_list.read().splitlines()

        print( "BatchLoader initialized with {} images".format(np.size(self.IndexList)))
    
    def load_next_tuple(self):
        if self._cur == (np.size(self.IndexList)-1):
            random.shuffle(self.IndexList)
            self._cur = 0

        # load data list
        data_list = self.all_data_list[(self.IndexList[self._cur])*26:(self.IndexList[self._cur])*26+26]
        
        # 4 load target mask image
        #tgt_mask = np.asarray(Image.open(data_list[11]))
        
        #tgt_mask = tgt_mask[:,:,3]
        #tgt_mask.setflags(write=1)
        
        #low_values_indices = tgt_mask < 128
        #tgt_mask[low_values_indices] = 0
        #high_values_indices = tgt_mask >= 128
        #tgt_mask[high_values_indices] = 1
        
        # 1 load RGB image
#        print(data_list)
        src_img = np.asarray(Image.open(data_list[0]))
        src_img = src_img[:,:,:3]
        src_img = np.rollaxis(src_img,2,0)        
        
        # 2 load depth map
        src_flowlabel = np.zeros((200,200))
        depth_map = np.zeros((1, src_flowlabel.shape[0], src_flowlabel.shape[1]))

        dm = np.fromfile(open(data_list[21]), dtype = np.double)
        dm.resize(src_flowlabel.shape[0],src_flowlabel.shape[1])
        depth_map[0,:,:] = dm.transpose()

        tgt_mask = np.zeros((200,200))
        tgt_mask[depth_map[0]>0] = 1

        # now finished! push the cursor
        self._cur = self._cur + 1

        return src_img, depth_map, tgt_mask


# borrow from python layers from Caffe Pascal Tutorial
def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['batch_size', 'data_list_dir', 'num_tform', 'image_size', 'data_num']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
