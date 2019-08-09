import numpy as np
import numba as nb
@nb.jit
def DepthMap2XyzImage(depth_map,K,Rt):
    
    P = np.dot(K,Rt)
#    C = np.dot(-(Rt[0:3,0:3]).transpose(), Rt[0:3,3])
    P_inv = np.linalg.pinv(P)
    C9 = np.dot(-(Rt[0:3,0:3]).transpose(), Rt[0:3,3])
    
    uv3 = np.array([100.5,100.5,1])
    xyz3 = np.dot(P_inv,uv3)
    xyz3 = xyz3 / xyz3[3]
    dir_C = xyz3[0:3] - C9
    dir_C = dir_C/np.linalg.norm(dir_C)    
    
    new_xyz_image = np.zeros((3,200,200))
    for xx in range(0,200):
        for yy in range(0,200):
            if depth_map[xx,yy] == 0:
                continue

            uv1 = np.array([yy+1,xx+1,1])
            xyz1 = np.dot(P_inv,uv1)
            xyz1 = xyz1 / xyz1[3]
            dir_ = xyz1[0:3] - C9
            dir_ = dir_ / np.linalg.norm(dir_)


            cos = np.dot(dir_,dir_C)

            depth = depth_map[xx,yy]/cos

            new_xyz = dir_*depth + C9
            new_xyz_image[:,xx,yy] = new_xyz
            
    return new_xyz_image
@nb.jit
def xyzImage2ForwardFlow(gt_xyz, K, Rt):
    P = np.dot(K,Rt)
    forward_flow = np.zeros((2,200,200))
    for x in range(0,200):
        for y in range(0,200):
            if np.sum(gt_xyz[:,x,y])==0:
                continue
            #if np.sum(gt_flow_F[:,x,y])==0:
            #    continue
            point = gt_xyz[:,x,y]
            
            point_1 = np.zeros(4)
            point_1[0:3] = point
            point_1[3] = 1
            
            UV = np.dot(P,point_1.transpose())
            UV = UV/UV[2]
            forward_flow[:,x,y] = UV[0:2] -1 -np.array([y,x])
            
    return forward_flow

def getKRtC(num):
    ex_mat = np.loadtxt('/home/lzy/data/PVHM_Tools/flow_renderer/extrinsic.txt')
    in_mat = np.loadtxt('/home/lzy/data/PVHM_Tools/flow_renderer/intrinsic.txt')
    K = in_mat[num*3:num*3+3,:]
    Rt = ex_mat[num*3:num*3+3,:]
    R = ex_mat[num*3:num*3+3,:3]
    t = ex_mat[num*3:num*3+3,3]
    C = -np.dot(R.transpose(),t)
    return K,Rt,C

def getKRtC_full_circle(num):
    ex_mat = np.loadtxt('/media/hao/mySpaceA/DataGeneratingChair/MatlabScript_2/DepthMapGenerate_front_200x200/extrinsic.txt')
    in_mat = np.loadtxt('/media/hao/mySpaceA/DataGeneratingChair/MatlabScript_2/DepthMapGenerate_front_200x200/intrinsic.txt')
    K = in_mat[num*3:num*3+3,:]
    Rt = ex_mat[num*3:num*3+3,:]
    R = ex_mat[num*3:num*3+3,:3]
    t = ex_mat[num*3:num*3+3,3]
    C = -np.dot(R.transpose(),t)
    return K,Rt,C


def getKRtCArray():
    ex_mat = np.loadtxt('/home/lzy/data/PVHM_Tools/flow_renderer/extrinsic.txt')
    in_mat = np.loadtxt('/home/lzy/data/PVHM_Tools/flow_renderer/intrinsic.txt')
    
    K = in_mat[0:3,:]

    Rt_arr = np.zeros((18,3,4))
    C_arr = np.zeros((18,3))
    for i in range(0,18):
        Rt_arr[i,:] = ex_mat[i*3:i*3+3,:]
        R = ex_mat[i*3:i*3+3,:3]
        t = ex_mat[i*3:i*3+3,3]
        C_arr[i,:] = -np.dot(R.transpose(),t)
    return K,Rt_arr,C_arr

def getKRtCArray_90():
    ex_mat = np.loadtxt('/media/hao/Data/DataSetGenerating9/MatlabScript/FlowMapGenerate3D_front_200x200_uvd_version/extrinsic_90_view.txt')
    in_mat = np.loadtxt('/media/hao/Data/DataSetGenerating9/MatlabScript/FlowMapGenerate_gt_uv_flow/intrinsic.txt')
    
    K = in_mat[0:3,:]

    Rt_arr = np.zeros((90,3,4))
    C_arr = np.zeros((90,3))
    for i in range(0,90):
        Rt_arr[i,:] = ex_mat[i*3:i*3+3,:]
        R = ex_mat[i*3:i*3+3,:3]
        t = ex_mat[i*3:i*3+3,3]
        C_arr[i,:] = -np.dot(R.transpose(),t)
    return K,Rt_arr,C_arr

def getKRtCArray_full_circle():
    ex_mat = np.loadtxt('/media/hao/mySpaceA/DataGeneratingChair/MatlabScript_2/DepthMapGenerate_front_200x200/extrinsic.txt')
    in_mat = np.loadtxt('/media/hao/mySpaceA/DataGeneratingChair/MatlabScript_2/DepthMapGenerate_front_200x200/intrinsic.txt')
    
    K = in_mat[0:3,:]

    Rt_arr = np.zeros((18,3,4))
    C_arr = np.zeros((18,3))
    for i in range(0,18):
        Rt_arr[i,:] = ex_mat[i*3:i*3+3,:]
        R = ex_mat[i*3:i*3+3,:3]
        t = ex_mat[i*3:i*3+3,3]
        C_arr[i,:] = -np.dot(R.transpose(),t)
    return K,Rt_arr,C_arr


def getBiasMat(height,width):
    bias_mat = np.zeros((2,height,width))
    for i in range(0,200):
        bias_mat[0,i,:] = np.array(range(0,height))
        bias_mat[1,:,i] = np.array(range(0,width)) 
    return bias_mat

