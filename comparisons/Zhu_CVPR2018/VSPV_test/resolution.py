import skimage
import numba as nb
import numpy as np
import scipy.misc
import scipy.ndimage
from skimage import data
from skimage.morphology import disk
from skimage.morphology import erosion
from skimage.filters.rank import median
import matplotlib.image as  mpimg

def getvalue(V,x,y,c,n,C,W,H):
    if ( x < 0 | x > W - 1 | y < 0 | y > H - 1 ) :
        return 0;
    return V[c,y,x];

@nb.jit
def remap(flow, srcImg):
    Vt = np.zeros(srcImg.shape)
    coords = flow
    C = srcImg.shape[0]
    W0 = srcImg.shape[1]
    H0 = srcImg.shape[2]
    C2 = flow.shape[0]
    W1 = flow.shape[1]
    H1 = flow.shape[2]
    
    n = 0
    
    if( W0!=W1 | H0!=H1  ):
#        print(W0,W1, H0,H1)
        print("Error: Input Sizes don't match.")
        return 0
    
    for c in range(0,C):
        for h in range(0,H1):
            for w in range(0,W1):
                x = coords[0,h,w];
                y = coords[1,h,w];
                x0 = np.floor(x);
                y0 = np.floor(y);
                x1 = x0 + 1;
                y1 = y0 + 1;
                wx = x - x0;
                wy = y - y0;
                w00 = (1 - wx) * (1 - wy);
                w01 = (1 - wx) * wy;
                w10 = wx * (1 - wy);
                w11 = wx * wy;
                
                if ( x0 < 0 or x0 > W1 - 1 or y0 < 0 or y0 > H1 - 1 ) :
                    continue
                if ( x0 < 0 or x0 > W1 - 1 or y1 < 0 or y1 > H1 - 1 ) :
                    continue
                if ( x1 < 0 or x1 > W1 - 1 or y0 < 0 or y0 > H1 - 1 ) :
                    continue
                if ( x1 < 0 or x1 > W1 - 1 or y1 < 0 or y1 > H1 - 1 ) :
                    continue
                
                v00 = srcImg[c,np.int(y0),np.int(x0)]
                v01 = srcImg[c,np.int(y1),np.int(x0)]
                v10 = srcImg[c,np.int(y0),np.int(x1)]
                v11 = srcImg[c,np.int(y1),np.int(x1)]
#                print(x0,x1,y0,y1)
                Vt[c,h,w] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                
    return Vt
    
def remap_resolution(flow, src_img_hr,resolution):
    flow_hr = np.zeros((2,resolution,resolution))
    flow_hr[0] = skimage.transform.resize(flow[0], (resolution,resolution))
    flow_hr[1] = skimage.transform.resize(flow[1], (resolution,resolution))
    flow_hr = 1.0*flow_hr*resolution/200

    mask = np.zeros((resolution,resolution))
    mask[np.sum(src_img_hr,0)>0] = 1
    mask = erosion(mask,disk(1))

    remaped_img_large = remap(flow_hr,src_img_hr)
    return remaped_img_large  







import matplotlib.pyplot as plt
import PIL
import os
import time

def test(left,right,mm,resolution,mid):
    for i in range(left,right):
        print(i)
        for j in range(18):    
            time1=time.time()                                  
            backward_flow=np.load('./backward_flow_200/{}_{}_{}_200.npy'.format(mm,i,j))[0]
#            print(backward_flow.shape)
#            print(np.max(backward_flow))
            
            srcImg=mpimg.imread('/home/lzy/data/PVHM_0-9999/rendered_images/{}_{}_{}x{}.png'.format(i,mid,resolution,resolution))[:,:,0:3]
#            print(np.max(srcImg))
            srcImg = np.rollaxis(srcImg,2,0)
            image=remap_resolution(backward_flow, srcImg,resolution)
            image=(255*image).astype(np.uint8)
            finalImg=np.copy(image)
#            print(np.max(image))
            
        #    print(image)
#            print(image.shape)
   

#            print(1111111111)
            image = np.moveaxis(image,0,2)
            image = PIL.Image.fromarray(((image))) 
        
            image.save('./image/{}_{}.png'.format(i,j))
            
            
            finalMask=np.load('./finalmask_200/{}_{}_{}.npy'.format(mm,i,j)) 
 #           print(finalMask.shape)
         
            
        
            finalMask_hr=np.zeros([resolution,resolution])
            
            finalMask_hr = skimage.transform.resize(finalMask, (resolution,resolution))
        
            
            finalImg[0][finalMask_hr==0] = 0
            finalImg[1][finalMask_hr==0] = 0
            finalImg[2][finalMask_hr==0] = 0
            
            finalImg=np.moveaxis(finalImg,0,2)
#            print(finalImg.shape)

                    
            image = PIL.Image.fromarray(((finalImg))) 
        
            
            
            
            
            path='./final_image_{}/{}-{}/{}'.format(resolution,left,right,i)
            folder=os.path.exists(path)
            if not folder:
                os.makedirs(path)    
                
            image.save(path+'/{}_{}.png'.format(mm,j))  
#            image.save('./200_{}.png'.format(j))  
            
            print(time1-time.time())
resolution=512

for mm in range(1,2):
    mid=9*mm
   
    test(9200,9400,mm,resolution,mid)                
