import numpy as np
import scipy.misc
import scipy.ndimage
from skimage import data
from skimage.morphology import disk
from skimage.morphology import erosion
from skimage.filters.rank import median

def getvalue(V,x,y,c,n,C,W,H):
    if ( x < 0 | x > W - 1 | y < 0 | y > H - 1 ) :
        return 0;
    return V[c,y,x];

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
    
    if( W0!=W1 | H0!=W1 | C != C2 ):
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
                
                Vt[c,h,w] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                
    return Vt
                
def remap_HR(flow, src_img_hr):
    flow_hr = np.zeros((2,500,500))
    flow_hr[0] = scipy.misc.imresize(flow[0], (500,500), interp = 'bicubic', mode = 'F')
    flow_hr[1] = scipy.misc.imresize(flow[1], (500,500), interp = 'bicubic', mode = 'F')
    flow_hr = flow_hr*2.5

    mask = np.zeros((500,500))
    mask[np.sum(src_img_hr,0)>0] = 1
    mask = erosion(mask,disk(1))
    src_img_hr_mf = np.zeros((3,500,500))
    src_img_hr_mf[0] = median(src_img_hr[0], disk(5), mask = mask)
    src_img_hr_mf[1] = median(src_img_hr[1], disk(5), mask = mask)
    src_img_hr_mf[2] = median(src_img_hr[2], disk(5), mask = mask)
    src_img_hr_mf[0][mask>0] = src_img_hr[0][mask>0]
    src_img_hr_mf[1][mask>0] = src_img_hr[1][mask>0]
    src_img_hr_mf[2][mask>0] = src_img_hr[2][mask>0]

    remaped_img_large = remap(flow_hr,src_img_hr_mf)
    return remaped_img_large

def remap_HR_with_errosion(flow, src_img_hr,er_num):
    flow_hr = np.zeros((2,500,500))
    flow_hr[0] = scipy.misc.imresize(flow[0], (500,500), interp = 'bicubic', mode = 'F')
    flow_hr[1] = scipy.misc.imresize(flow[1], (500,500), interp = 'bicubic', mode = 'F')
    flow_hr = flow_hr*2.5

    mask = np.zeros((500,500))
    mask[np.sum(src_img_hr,0)>0] = 1
    mask = erosion(mask,disk(er_num))
    src_img_hr_mf = np.zeros((3,500,500))
    src_img_hr_mf[0] = median(src_img_hr[0], disk(5), mask = mask)
    src_img_hr_mf[1] = median(src_img_hr[1], disk(5), mask = mask)
    src_img_hr_mf[2] = median(src_img_hr[2], disk(5), mask = mask)
    src_img_hr_mf[0][mask>0] = src_img_hr[0][mask>0]
    src_img_hr_mf[1][mask>0] = src_img_hr[1][mask>0]
    src_img_hr_mf[2][mask>0] = src_img_hr[2][mask>0]

    remaped_img_large = remap(flow_hr,src_img_hr_mf)
    return remaped_img_large

def remap_HR_sparse(flow, src_img_hr):
    flow_hr = np.zeros((2,500,500))
    for x in range(0,200):
        for y in range(0,200):
            if flow[0][x][y] != 0:
                flow_hr[0][np.int(x*2.5+0.5)][np.int(y*2.5+0.5)] = 2.5*flow[0][x][y]
                flow_hr[1][np.int(x*2.5+0.5)][np.int(y*2.5+0.5)] = 2.5*flow[1][x][y]
    flow_hr[flow_hr==0] = -500

    remaped_img_large = remap(flow_hr,src_img_hr)
    return remaped_img_large

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
