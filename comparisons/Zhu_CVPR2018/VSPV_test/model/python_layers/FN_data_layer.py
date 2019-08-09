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
import matplotlib.image as  mpimg
import random
import PIL

class FN_data_layer(caffe.Layer):
    
    def setup(self, bottom, top):
        
        self.top_names = ['srcImg', 'srcFlow', 'tform', 'tgtImg', 'tgtMask']
        
        # analysis pamameters
        params = eval(self.param_str)
        
        check_params(params)

        self.batch_size = params['batch_size']
        
        self.batch_loader = BatchLoader(params, None)
        
        top[0].reshape(self.batch_size, 2, params['image_size'][0], params['image_size'][1])
        top[1].reshape(self.batch_size, 2, params['image_size'][0], params['image_size'][1])
        top[2].reshape(self.batch_size, 1, params['image_size'][0], params['image_size'][1])
        top[3].reshape(self.batch_size, 3, params['image_size'][0], params['image_size'][1])
        top[4].reshape(self.batch_size, 3, params['image_size'][0], params['image_size'][1])
        
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            gt_flow_F, gt_flow_B, tgt_mask, src_img, tgt_img = \
            self.batch_loader.load_next_tuple()

            top[0].data[itt, ...] = gt_flow_F.astype(np.float)
            top[1].data[itt, ...] = gt_flow_B.astype(np.float)
            top[2].data[itt, 0, ...] = tgt_mask.astype(np.float)
            top[3].data[itt, ...] = src_img.astype(np.float)
            top[4].data[itt, ...] = tgt_img.astype(np.float)
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
        self.cal=0
        
        self.bias_mat = np.zeros((2,200,200))
        for i in range(0,200):
            self.bias_mat[0,i,:] = np.array(range(0,200))
            self.bias_mat[1,:,i] = np.array(range(0,200))        


        # make index list
        self.IndexList = []
        for i_ind in range(0,self.data_num):
            self.IndexList.append(i_ind)
            #if (i_ind%18==4):
            #    self.IndexList.append(i_ind)

        # shuffle the index list
#        random.shuffle(self.IndexList)

        # read All Tuple List (26 as a batch)
        f_all_data_list = open(self.data_list_dir)
        self.all_data_list = f_all_data_list.read().splitlines()
        
        print( "BatchLoader initialized with {} images".format(np.size(self.IndexList)))
    
    def load_next_tuple(self):
        if self._cur == (np.size(self.IndexList)-1):
            random.shuffle(self.IndexList)
            self._cur = 0

        # load data list
        print(self._cur)
        data_list = self.all_data_list[(self.IndexList[self._cur])*26:(self.IndexList[self._cur])*26+26]
        
        # 4 load target mask image
#        tgt_mask = np.asarray(Image.open(data_list[3]))
        tgt_mask=(mpimg.imread(data_list[3])*255).astype(int)
        tgt_mask = tgt_mask[:,:,3]
        tgt_mask.setflags(write=1)
        
        low_values_indices = tgt_mask < 10
        tgt_mask[low_values_indices] = 0
        high_values_indices = tgt_mask >= 10
        tgt_mask[high_values_indices] = 1
        

                
        
        # load remaped forward flow
        gt_flow_F = np.zeros((2,200,200))
        
#        flow_u = np.fromfile(open(data_list[24]), dtype = np.double)
        flow_u=np.load(data_list[24])
        flow_u.resize(200,200)
        gt_flow_F[0,:,:] = flow_u.transpose()
#        print(np.max(flow_u))
#        flow_v = np.fromfile(open(data_list[25]), dtype = np.double)
        flow_v=np.load(data_list[25])

        flow_v.resize(200,200)
        gt_flow_F[1,:,:] = flow_v.transpose()


        # load ground truth flow (backward)
        gt_flow_B = np.zeros((2,200,200))
        
        flow_u = np.fromfile(open(data_list[5]), dtype = np.double)
#        print(np.max(flow_u))
        flow_u.resize(200,200)
        gt_flow_B[0,:,:] = flow_u.transpose()

        flow_v = np.fromfile(open(data_list[6]), dtype = np.double)
        flow_v.resize(200,200)
        gt_flow_B[1,:,:] = flow_v.transpose()
#        print(data_list[1])
#        print(data_list[4])
        tgt_mask = np.zeros((200,200))
        tgt_mask[np.sum(gt_flow_B,0)!=0] = 1
        

        # 1 load RGB image
        src_img = np.asarray(Image.open(data_list[0]))
        src_img = src_img[:,:,:3]
        src_img = np.rollaxis(src_img,2,0)
        
        # 3 load target RGB image
#        tgt_img = np.asarray(Image.open(data_list[2]))
        tgt_img=(mpimg.imread(data_list[2])*255).astype(int)
        tgt_img = tgt_img[:,:,:3]
        tgt_img = np.rollaxis(tgt_img,2,0)

        tgt_img.setflags(write=1)
        tgt_img[0,:,:] = tgt_img[0,:,:] * tgt_mask
        tgt_img[1,:,:] = tgt_img[1,:,:] * tgt_mask
        tgt_img[2,:,:] = tgt_img[2,:,:] * tgt_mask


        # now finished! push the cursor
        self._cur = self._cur + 1
        image = PIL.Image.fromarray(((np.moveaxis(tgt_img,0,2).astype(np.uint8)))) 
        image.save('./image/{}_2.png'.format(self.cal))
        
        mask1=np.zeros((200,200))
        for i in range(200):
            for j in range(200):
                if gt_flow_F[0,i,j]!=0:
                    mask1[i,j]=255               
        image = PIL.Image.fromarray(((mask1.astype(np.uint8)))) 
        image.save('./mask/{}_3.png'.format(self.cal))
        
        mask1=np.zeros((200,200))
        for i in range(200):
            for j in range(200):
                if gt_flow_B[0,i,j]!=0:
                    mask1[i,j]=255
        image = PIL.Image.fromarray(((mask1.astype(np.uint8)))) 
        image.save('./mask/{}_4.png'.format(self.cal))
                    
#        print(src_img.shape)       
        image = PIL.Image.fromarray(((np.moveaxis(src_img,0,2).astype(np.uint8)))) 
        image.save('./image/{}_1.png'.format(self.cal))
        
        np.save('./srcImg/{}_2.npy'.format(self.cal),src_img)
        np.save('./forward_flow/{}_2.npy'.format(self.cal),gt_flow_F)
        print(data_list[24])
        print(data_list[25])
#        print(src_img.shape)
        print(data_list[1],data_list[4],self._cur)
        self.cal=self.cal+1
        return gt_flow_F, gt_flow_B, tgt_mask, src_img, tgt_img


# borrow from python layers from Caffe Pascal Tutorial
def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['batch_size', 'data_list_dir', 'num_tform', 'image_size', 'data_num']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
