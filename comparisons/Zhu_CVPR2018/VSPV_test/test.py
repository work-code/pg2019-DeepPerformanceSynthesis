import numpy as np
import matplotlib.pyplot as plt
import PIL
import scipy
import skimage
from skimage.morphology import disk
import sys
sys.path.append('./utility/')
from show_functions import display_img_array
from project_functions import DepthMap2XyzImage
from project_functions import xyzImage2ForwardFlow
from project_functions import getKRtCArray
from project_functions import getBiasMat
from remap_functions import remap

#import pycaffe
sys.path.append('./caffe_VSPV/python/')
import caffe

caffe.set_mode_gpu()

sys.path.append("./model/python_layers/")
DN = caffe.Net('./model/DepthNet_deploy.prototxt','./snapshot/DN/iter_iter_200000.caffemodel', caffe.TEST)
FN = caffe.Net('./model/FlowNet_deploy.prototxt','./snapshot/FN/iter_iter_170000.caffemodel', caffe.TEST)
MN = caffe.Net('./model/MaskNet_deploy.prototxt','./snapshot/MN/iter_iter_100000.caffemodel', caffe.TEST)

K, Rt_arr, C_arr = getKRtCArray()


from skimage.morphology import disk

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


# In[124]:


num_tform = 5
test_num = 5860

src_mask = np.asarray(PIL.Image.open('/media/hao/Data/DataSetGenerating9/image_200x200/{}_9_m.png'.format(test_num)))
src_mask = src_mask[:,:,3]
src_mask.setflags(write=1)
low_values_indices = src_mask < 128
src_mask[low_values_indices] = 0
high_values_indices = src_mask >= 128
src_mask[high_values_indices] = 1

# 1 load RGB image
src_img = np.asarray(PIL.Image.open('/media/hao/Data/DataSetGenerating9/image_200x200/{}_9_c.png'.format(test_num)))
src_img = src_img[:,:,:3]
src_img = np.rollaxis(src_img,2,0)

src_img_hr = np.asarray(PIL.Image.open('/media/hao/Data/DataSetGenerating9/image_500x500/{}_9_c.png'.format(test_num)))
src_img_hr = src_img_hr[:,:,:3]
src_img_hr = np.rollaxis(src_img_hr,2,0)

tform_vec = np.zeros(18)
tform_vec[num_tform] = 1

tgt_img = np.asarray(PIL.Image.open('/media/hao/Data/DataSetGenerating9/image_200x200/{}_{}_c.png'.format(test_num,num_tform)))
tgt_img = tgt_img[:,:,:3]
tgt_img = np.rollaxis(tgt_img,2,0)

DN.blobs['srcImg'].data[0] = src_img
DN.forward()
predDepth = DN.blobs['final'].data[0,0]
predDepth = predDepth * src_mask

# depth 2 forward flow
xyz_image= DepthMap2XyzImage(predDepth, K, Rt_arr[9])
forward_flow = xyzImage2ForwardFlow(xyz_image, K, Rt_arr[num_tform])

FN.blobs['forward_flow'].data[0] = forward_flow
FN.blobs['gt_xyz'].data[0] = xyz_image
FN.blobs['srcImg'].data[0] = src_img
FN.forward()
predImg = FN.blobs['predImg'].data[0]

flow_mask = forward2backwardMap(forward_flow)
flow_mask = skimage.morphology.closing(flow_mask,disk(2)) * 255
flow_mask = flow_mask

forward_flow[0][src_mask==0] = 0
forward_flow[1][src_mask==0] = 0

MN.blobs['srcMask'].data[0,0] = src_mask
MN.blobs['srcFlow'].data[0] = forward_flow
MN.blobs['tform'].data[0] = tform_vec

MN.forward()
predMask = MN.blobs['predMask'].data[0,0]

finalMask = predMask.astype(np.float) + flow_mask.astype(np.float)
finalMask[finalMask>0] = 255
finalImg = np.copy(predImg)
finalImg[0][finalMask==0] = 0
finalImg[1][finalMask==0] = 0
finalImg[2][finalMask==0] = 0
display_img_array((np.moveaxis(finalImg,0,2)).astype(np.uint8))

