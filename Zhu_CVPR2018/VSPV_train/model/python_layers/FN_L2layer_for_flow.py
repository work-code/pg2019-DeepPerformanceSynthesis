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

class FN_L2layer_for_flow(caffe.Layer):
    
    def setup(self, bottom, top):

        #self.param_ = MaskedL1Loss.parse_args(self.param_str)

        params = eval(self.param_str)
        self.loss_weight = params['loss_weight']

        assert len(bottom) == 3, 'there should be 3 bottom blobs: predFlow, gtFlow '
        
        self.batch_num = bottom[0].data.shape[0]
        self.batchSz_ = bottom[0].num
        
    def reshape(self, bottom, top):
        top[0].reshape(1)
        pass

    def forward(self, bottom, top):
        
        # before computing loss, mask the predImage
        predFlow_t = bottom[0]
        for i_batch in range(predFlow_t.data.shape[0]):
            for i_channel in range(predFlow_t.data.shape[1]):
                predFlow_t.data[i_batch][i_channel][...] = predFlow_t.data[i_batch][i_channel][...] * bottom[2].data[i_batch][0]
        
        gtFlow_t = bottom[1]
        for i_batch in range(gtFlow_t.data.shape[0]):
            for i_channel in range(gtFlow_t.data.shape[1]):
                gtFlow_t.data[i_batch][i_channel][...] = gtFlow_t.data[i_batch][i_channel][...] * bottom[2].data[i_batch][0]
        
        loss, count = 0, 0
        for b in range(self.batchSz_):
            err   = predFlow_t.data[b].squeeze() - gtFlow_t.data[b].squeeze()
            err   = np.array(err)
            loss  += 0.5 * np.sum(err * err)
            count    += 1
        if count == 0:
            top[0].data[...] = 0.0
        else:
            top[0].data[...] = self.loss_weight * loss /float(count)
        
    def backward(self, top, propagate_down, bottom):
        predFlow_t = bottom[0]
        for i_batch in range(predFlow_t.data.shape[0]):
            for i_channel in range(predFlow_t.data.shape[1]):
                predFlow_t.data[i_batch][i_channel][...] = predFlow_t.data[i_batch][i_channel][...] * bottom[2].data[i_batch][0]
        
        gtFlow_t = bottom[1]
        for i_batch in range(gtFlow_t.data.shape[0]):
            for i_channel in range(gtFlow_t.data.shape[1]):
                gtFlow_t.data[i_batch][i_channel][...] = gtFlow_t.data[i_batch][i_channel][...] * bottom[2].data[i_batch][0]
        
        count = 0
        bottom[0].diff[...] = 0
        for b in range(self.batchSz_):
            count += 1
            diff   = predFlow_t.data[b].squeeze() - gtFlow_t.data[b].squeeze()
            bottom[0].diff[b] = diff[...].reshape(bottom[0].diff[b].shape)
        if count == 0:
            bottom[0].diff[...] = 0
        else:
            bottom[0].diff[...] = self.loss_weight * bottom[0].diff[...]/float(count)
