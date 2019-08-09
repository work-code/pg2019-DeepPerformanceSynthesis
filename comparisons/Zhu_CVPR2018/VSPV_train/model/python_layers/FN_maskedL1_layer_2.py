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
import PIL
import random

class FN_maskedL1_layer_2(caffe.Layer):
    
    def setup(self, bottom, top):

        #self.param_ = MaskedL1Loss.parse_args(self.param_str)

        self.top_names = ['imgLoss']
        self.bottom_names = ['PredImg', 'tgtImg', 'imgLoss']
        
        params = eval(self.param_str)
        self.loss_weight = params['loss_weight']

        assert len(bottom) == 2, 'there should be two bottom blobs: predImg, tgtImg'
        
        self.predShape = bottom[0].data.shape
        self.tgtShape = bottom[1].data.shape
        
        self.cal=0
        
        self.i=900
        self.j=0

    def reshape(self, bottom, top):
        top[0].reshape(1,1,1,1)
        pass

    def forward(self, bottom, top):
       
        # before computing loss, mask the predImage
        predImage_t = np.copy(bottom[0].data)
#        image = PIL.Image.fromarray(((np.moveaxis(predImage_t[0,:,:,:],0,2).astype(np.uint8)))) 
#        image.save('./FN_image/{}_1.png'.format(self.cal))
#        image = PIL.Image.fromarray(((np.moveaxis(bottom[1].data[0,:,:,:],0,2).astype(np.uint8)))) 
#        image.save('./FN_image/{}_2.png'.format(self.cal))
#        image = PIL.Image.fromarray(((np.moveaxis(predImage_t[0,:,:,:],0,2).astype(np.uint8)))) 
#        image.save('./image/{}_1.png'.format(self.cal))
        
#        for i in range(predImage_t.shape[0]):
#            image = PIL.Image.fromarray(((np.moveaxis(predImage_t[i,:,:,:],0,2).astype(np.uint8)))) 
#            image.save('/home/lzy/data/train_pred_images/{}_{}.png'.format(self.i,self.j))
#            self.j=self.j+1
#            if self.j==18:
#                self.j=0
#                self.i=self.i+1
        for i_batch in range(predImage_t.shape[0]):
            for i_channel in range(predImage_t.shape[1]):
                mask = np.zeros((200,200))
                mask[np.sum(bottom[1].data[i_batch],0)>0] = 1
                predImage_t[i_batch][i_channel][...] = predImage_t[i_batch][i_channel][...] * mask


        top[0].data[...] = self.loss_weight * np.sum(np.abs(predImage_t[...].squeeze() - bottom[1].data[...].squeeze()))/(float(self.tgtShape[0]))
#        print(predImage_t[...].shape)
#        image = PIL.Image.fromarray(((np.moveaxis(predImage_t[0,:,:,:],0,2).astype(np.uint8)))) 
#        image.save('./FN_image/{}_3.png'.format(self.cal))
#        image = PIL.Image.fromarray(((np.moveaxis(predImage_t[1,:,:,:],0,2).astype(np.uint8)))) 
#        image.save('./image/{}_4_1.png'.format(self.cal))
#        image = PIL.Image.fromarray(((np.moveaxis(bottom[1].data[0,:,:,:],0,2).astype(np.uint8)))) 
#        image.save('./FN_image/{}_4.png'.format(self.cal))
#        mask1=np.zeros((200,200))
#        mask2=np.zeros((200,200))
#        for i in range(200):
#            for j in range(200):
#                if predImage_t[0,0,i,j]!=0:
#                    mask1[i,j]=255
#                if bottom[1].data[0,0,i,j]!=0:
#                    mask2[i,j]=255
#        image = PIL.Image.fromarray(((mask1.astype(np.uint8)))) 
#        image.save('./FN_image_mask/{}_1.png'.format(self.cal))
#        image = PIL.Image.fromarray(((mask2.astype(np.uint8)))) 
#        image.save('./FN_image_mask/{}_2.png'.format(self.cal))
        self.cal=self.cal+1
    def backward(self, top, propagate_down, bottom):



        # before computing loss, mask the predImage
        predImage_t = np.copy(bottom[0].data)
        for i_batch in range(predImage_t.shape[0]):
            for i_channel in range(predImage_t.shape[1]):
                mask = np.zeros((200,200))
                mask[np.sum(bottom[1].data[i_batch],0)>0] = 1
                predImage_t[i_batch][i_channel][...] = predImage_t[i_batch][i_channel][...] * mask

        bottom[0].diff[...] = self.loss_weight * np.sign(predImage_t[...].squeeze() - bottom[1].data[...].squeeze())/(float(self.tgtShape[0]))
