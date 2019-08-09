
#coding: utf-8

# # train mask net

# In[ ]:

#
#import caffe
#import numpy as np
#import sys
#caffe_root = './caffe_VSPV/'
#sys.path.append(caffe_root + 'python')
#sys.path.append("./model/python_layers/")
#
#
#
#caffe.set_mode_gpu()
#
## read solver
#solver = caffe.AdamSolver('./solver/mySolver_MaskNet.prototxt')
#
#solver.restore('./snapshot/MN/iter_iter_20000.solverstate')
#MN_loss=0
#for iter in range(200000):
#    solver.step(1)
#    if iter%1000 == 0:
#        MN_loss=MN_loss+solver.net.blobs['maskLoss'].data[0,0,0,0]
##        MN_loss=open('MN_loss.txt','a')
##        print( "{} {}".format(solver.iter,solver.net.blobs['maskLoss'].data[0,0,0,0]))
#        print( "{} {}".format(solver.iter,MN_loss))
#        MN_loss=0
##        MN_loss.write(str(solver.net.blobs['maskLoss'].data[0,0,0,0]))    
##        MN_loss.write('\n')
##        MN_loss.close()
##
# train flow net

# In[ ]:


import caffe
import numpy as np
import sys
caffe_root = './caffe_VSPV/'
sys.path.append(caffe_root + 'python')
sys.path.append("./model/python_layers/") 



caffe.set_mode_gpu()

# read solver
solver = caffe.AdamSolver('./solver/mySolver_FlowNet.prototxt')

solver.restore('./snapshot/FN/iter_iter_136000.solverstate')

# small model
# L1 flow loss + 0.01 L1 img loss
flowL1Loss_sum=0
imgLoss_sum=0
for iter in range(100000):
    solver.step(1)
    flowL1Loss_sum=flowL1Loss_sum+solver.net.blobs['flowL1Loss'].data[0,0,0,0]
    imgLoss_sum=imgLoss_sum+solver.net.blobs['imgLoss'].data[0,0,0,0]
    FN_loss=open('FN_loss_3.txt','a')
    FN_loss.write(str(solver.net.blobs['flowL1Loss'].data[0,0,0,0])+'\t'+str(solver.net.blobs['imgLoss'].data[0,0,0,0]))    
    FN_loss.write('\n')
    FN_loss.close()
    if iter%1000 == 999:        
        FN_loss_sum=open('FN_loss_sum_3.txt','a')
#        print ("{} {} {}".format(solver.iter,solver.net.blobs['flowL1Loss'].data[0,0,0,0],solver.net.blobs['imgLoss'].data[0,0,0,0]))
        print ("{} {} {}".format(solver.iter,flowL1Loss_sum,imgLoss_sum))       
        FN_loss_sum.write(str(flowL1Loss_sum)+'\t'+str(imgLoss_sum))    
        FN_loss_sum.write('\n')
        FN_loss_sum.close()
        flowL1Loss_sum=0
        imgLoss_sum=0
#
# # train depth net

# In[ ]:
#
#
#import caffe
#import numpy as np
#import sys
#import os
#caffe_root = './caffe_VSPV/'
#sys.path.append(caffe_root + 'python')
#sys.path.append("./model/python_layers/")
#
#
#print(os.getcwd())
#caffe.set_mode_gpu()
#np.save('1.txt',np.zeros(3))
## read solver
#solver = caffe.AdamSolver('./solver/mySolver_DepthNet.prototxt')
#
#solver.restore('./snapshot/DN/iter_iter_55000.solverstate')
#
##solver.restore('./snapshot/snapshot_mySolver_iter_150000.solverstate')
#DN_loss=0
## from scratch test
#for iter in range(50000):
#    solver.step(1)
#    DN_loss=DN_loss+solver.net.blobs['loss'].data[0,0,0,0]
#    if iter%1000 == 0:
##        print( "{} {}".format(solver.iter,solver.net.blobs['loss'].data[0,0,0,0]))
#        print  ( "{} {}".format(solver.iter,DN_loss))
#        DN_loss=0
