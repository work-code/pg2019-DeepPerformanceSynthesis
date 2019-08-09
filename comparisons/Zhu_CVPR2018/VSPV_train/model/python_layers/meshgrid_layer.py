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

class meshgrid_layer(caffe.Layer):
    
    def setup(self, bottom, top):
        
        #print 'here is the setup of Meshgrid'

        self.top_names = ['gxy']

        params = eval(self.param_str)

        self.height = params['height']
        self.width = params['width']
        self.batch_size = params['batch_size']

        top[0].reshape(self.batch_size, 2, self.height, self.width)
        
        #print 'here is the setup of Meshgrid'

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        
        img_xy = np.zeros((2, self.height, self.width))

        for iter_h in range(self.height):
            for iter_w in range(self.width):
                img_xy[0][iter_h][iter_w] = iter_w
                img_xy[1][iter_h][iter_w] = iter_h

        for iter_b in range(self.batch_size):
            top[0].data[iter_b, ...] = img_xy


    def backward(self, top, propagate_down, bottom):
        pass
