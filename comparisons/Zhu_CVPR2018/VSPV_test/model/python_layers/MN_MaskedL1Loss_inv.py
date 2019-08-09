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
import PIL

class MN_MaskedL1Loss_inv(caffe.Layer):
    
    def setup(self, bottom, top):

        #self.param_ = MaskedL1Loss.parse_args(self.param_str)

        self.top_names = ['imgLoss']
        self.bottom_names = ['PredImg', 'tgtImg', 'imgLoss']
        
        params = eval(self.param_str)
        self.loss_weight = params['loss_weight']

        assert len(bottom) == 3, 'there should be three bottom blobs: predImg, tgtImg, tgtMask '
        
        self.predShape = bottom[0].data.shape
        self.tgtShape = bottom[1].data.shape
        self.tgtMask = bottom[2].data.shape
        self.cal=0
        self.i=900
        self.j=0
    def reshape(self, bottom, top):
        top[0].reshape(1,1,1,1)
        pass

    def forward(self, bottom, top):
       
        # before computing loss, mask the predImage
        predImage_t = bottom[0]
        image = PIL.Image.fromarray(((predImage_t.data[0,0,:,:].astype(np.uint8)))) 
        image.save('./MN_mask/{}_2.png'.format(self.cal))
        image = PIL.Image.fromarray(((bottom[1].data[0,0,:,:].astype(np.uint8)))) 
        image.save('./MN_mask/{}_1.png'.format(self.cal))
        for i in range(predImage_t.data.shape[0]):
            print(predImage_t.data[i,:,:,:].shape)
#            image = PIL.Image.fromarray(((predImage_t.data[i,0,:,:].astype(np.uint8)))) 
            
            np.save('/home/lzy/data/train_pre_mask/{}_{}.png'.format(self.i,self.j),predImage_t.data[i,0,:,:])
            self.j=self.j+1
            if self.j==18:
                self.j=0
                self.i=self.i+1
        
        for i_batch in range(predImage_t.data.shape[0]):
            mask = np.zeros((200,200))
            mask[bottom[2].data[i_batch][0]==0] = 1
            for i_channel in range(predImage_t.data.shape[1]):
                predImage_t.data[i_batch][i_channel][...] = predImage_t.data[i_batch][i_channel][...] * mask
                bottom[1].data[i_batch][i_channel][...] = bottom[1].data[i_batch][i_channel][...] * mask

        top[0].data[...] = self.loss_weight * np.sum(np.abs(predImage_t.data[...].squeeze() - bottom[1].data[...].squeeze()))/(float(self.tgtShape[0]))
        print(np.max(bottom[1].data[...][0,:,:,:]),bottom[1].data[...].shape)
      
  
        self.cal=self.cal+1
    def backward(self, top, propagate_down, bottom):



        # before computing loss, mask the predImage
        predImage_t = bottom[0]
        for i_batch in range(predImage_t.data.shape[0]):
            mask = np.zeros((200,200))
            mask[bottom[2].data[i_batch][0]==0] = 1
            for i_channel in range(predImage_t.data.shape[1]):
                predImage_t.data[i_batch][i_channel][...] = predImage_t.data[i_batch][i_channel][...] * mask
                bottom[1].data[i_batch][i_channel][...] = bottom[1].data[i_batch][i_channel][...] * mask

        bottom[0].diff[...] = self.loss_weight * np.sign(predImage_t.data[...] - bottom[1].data[...])/(float(self.tgtShape[0]))
