import numpy as np
import math
import time
import torch
import gc
import numba as nb
#
#@nb.vectorize(nopython=True)
#@nb.jit(nopython=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def crossing_s(a,b,number):
    c=torch.empty((number,a.shape[0],3)).cuda()   
    c[:,:,0]=a[:,1]*b[:,2]-a[:,2]*b[:,1]
    c[:,:,1]=a[:,2]*b[:,0]-a[:,0]*b[:,2]
    c[:,:,2]=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    return c

def crossing_t(a,b,number):
#    print(a.shape,b.shape)
    c=torch.empty((number,a.shape[0],3)).cuda()   
#    print((a[:,1]*b[:,:,2]-a[:,2]*b[:,:,1]).shape)
    c[:,:,0]=a[:,1]*b[:,:,2]-a[:,2]*b[:,:,1]
    c[:,:,1]=a[:,2]*b[:,:,0]-a[:,0]*b[:,:,2]
    c[:,:,2]=a[:,0]*b[:,:,1]-a[:,1]*b[:,:,0]
    return c

@nb.jit(nopython=True)
def crossing(a,b):  
    c=np.zeros(3)
    c[0]=a[1]*b[2]-a[2]*b[1]
    c[1]=a[2]*b[0]-a[0]*b[2]
    c[2]=a[0]*b[1]-a[1]*b[0]
    return c
#@nb.jit(nopython=True)
#@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])],'(n),(n)->(n)')
#def sub(a,b,c):  
#    for i in range(3):
#        c[i]=a[i]-b[i]
      

#@nb.jit(nopython=True)
#def equal(a,b):
##    print(a,b)
#    if a[0]*b[0]+a[1]*b[1]+a[2]*b[2]<0:
#        
#        return False
#    else:
##        print(1111)
#        return True
    
#    a=a/np.sqrt(dot(a[0],0,a[1],0,a[2],0))
#    b=b/np.sqrt(dot(b[0],0,b[1],0,b[2],0))
#    for i in range(3):
#        if math.fabs(a[i]-b[i])>0.00000001:
#            return False
#    return True  
  
@nb.jit(nopython=True)
def length_to_edge(a,b,point):
#    print(a,b,point)
#    if np.dot(b-a,point-a)<0:
#        return dot(point[0],a[0],point[1],a[1],point[2],a[2])
#    elif np.dot(a-b,point-b)<0:
##        return np.sqrt(dot(point[0],b[0],point[1],b[1],point[2],b[2]))
#        return dot(point[0],b[0],point[1],b[1],point[2],b[2])
    c=crossing(point-a,point-b)
    d=b-a
    return dot(c[0],0,c[1],0,c[2],0)/dot(d[0],0,d[1],0,d[2],0)


def length(number,triangles,points):
    triangles=torch.tensor(triangles)
    points=torch.tensor(points)
    
    triangles=triangles.cuda()
#    
    points=points.cuda()   
    
    time1=time.time()
    points=points.type(torch.float)
    triangles=triangles.type(torch.float)
    a=triangles[:,1,:]-triangles[:,0,:]
    b=triangles[:,2,:]-triangles[:,0,:]
    c=crossing_s(a,b,points.shape[0])


    m=torch.zeros(points.shape[0],number).type(torch.float).cuda()   
    for i in range(3):
       m=m+triangles[:,0,i]*c[:,:,i]
    length1=((c[:,:,0].transpose(0,1)*points[:,0]+c[:,:,1].transpose(0,1)*points[:,1] +c[:,:,2].transpose(0,1)*points[:,2]-m.transpose(0,1)).transpose(0,1))/torch.sqrt(c[:,:,0]*c[:,:,0]+c[:,:,1]*c[:,:,1]+c[:,:,2]*c[:,:,2]).type(torch.float)  
    k=((c[:,:,0].transpose(0,1)*points[:,0]+c[:,:,1].transpose(0,1)*points[:,1] +c[:,:,2].transpose(0,1)*points[:,2]-m.transpose(0,1)).transpose(0,1))/(c[:,:,0]*c[:,:,0]+c[:,:,1]*c[:,:,1]+c[:,:,2]*c[:,:,2])  
#   
#    print(time.time()-time1)
#    
    d=torch.empty((triangles.shape[0],3)).cuda()
    for i in range(3):
        d[:,i]=triangles[:,2,i]-triangles[:,1,i]  
    triangles_points= torch.empty((points.shape[0],triangles.shape[0],3,3)).cuda()  
    triangles=triangles.expand(points.shape[0],number,3,3).cuda()
    for i in range(3):
        for j in range(3):
            triangles_points[:,:,i,j]=triangles[:,:,i,j]+k*c[:,:,j]  
        
    vector=(points.transpose(0,1)-triangles_points.transpose(0,3).transpose(0,2)).transpose(0,2).transpose(0,3)

    cross=torch.empty((points.shape[0],number,3,3)).cuda()    
    cross[:,:,0,:]=crossing_t(a,vector[:,:,0,:],points.shape[0])
    cross[:,:,1,:]=crossing_t(d,vector[:,:,1,:],points.shape[0])
    cross[:,:,2,:]=crossing_t(-b,vector[:,:,2,:],points.shape[0])
    
#    print(torch.cuda.memory_allocated())
    del vector,a,b,d,triangles_points,k,m
    length1=length1.cpu().numpy().astype(float)  
    torch.cuda.empty_cache()
#    print(torch.cuda.memory_allocated())
#    gc.collect()
#    print(torch.cuda.memory_allocated())
       
    
    cross_dot=torch.empty((points.shape[0],number,3)).cuda()  
    for i in range(3):
        cross_dot[:,:,i]=-(cross[:,:,i,0]*c[:,:,0]+cross[:,:,i,1]*c[:,:,1]+cross[:,:,i,2]*c[:,:,2])
    cross_dot=cross_dot.cpu().numpy().astype(float)   
    torch.cuda.empty_cache()   
    
    distance=torch.empty((points.shape[0],number,3))  
    for i in range(3):
        distance[:,:,i]=torch.norm((points.transpose(0,1)-triangles[:,:,i,:].transpose(0,2).transpose(0,1)).transpose(0,1).transpose(0,2),2,2)
#   
    distance_max=torch.empty(points.shape[0],number)
    distance_max=torch.max(distance,2)[0]   
#    print(torch.cuda.memory_allocated())     
    del distance
    distance_max=(distance_max.cpu()).numpy().astype(float)  
    torch.cuda.empty_cache()
#    print(torch.cuda.memory_allocated())
    
    vector=torch.empty((points.shape[0],number,3,3)).cuda()
    for i in range(3):
        for j in range(3):
            vector[:,:,i,j]=(triangles[:,:,i,0]-triangles[:,:,j,0])*(points[:,0]-triangles[:,:,j,0].transpose(0,1)).transpose(0,1)+(triangles[:,:,i,1]-triangles[:,:,j,1])*(points[:,1]-triangles[:,:,j,1].transpose(0,1)).transpose(0,1)+(triangles[:,:,i,2]-triangles[:,:,j,2])*(points[:,2]-triangles[:,:,j,2].transpose(0,1)).transpose(0,1)

              
    points=points.cpu().numpy()
    triangles=(triangles.cpu().numpy()).astype(float)  
    vector=(vector.cpu()).numpy().astype(float)  
   
    

    min_=length_min(cross_dot,length1,number,points,triangles,distance_max,vector)
     
#    min_=0
    return min_
 
@nb.jit(nopython=True)
def length_min(cross_dot,length1,number,point,triangles,distance_max,vector):
#     min_=10000.0 

     min_=np.empty(cross_dot.shape[0])
     print(min_.shape)
     for j in range(cross_dot.shape[0]):
         min_[j]=10000.0
         for i in range(number): 
             
            if cross_dot[j,i,0]>0 and vector[j,i,1,0]>0 and vector[j,i,0,1]>0:            
                length2=length_to_edge(triangles[j,i,0,:],triangles[j,i,1,:],point[j,:])
            elif cross_dot[j,i,1]>0  and vector[j,i,2,1] and vector[j,i,1,1]>0:
                length2=length_to_edge(triangles[j,i,1,:],triangles[j,i,2,:],point[j,:])
            elif cross_dot[j,i,2]>0 and vector[j,i,2,0]>0 and vector[j,i,0,2]>0:
                length2=length_to_edge(triangles[j,i,0,:],triangles[j,i,2,:],point[j,:])
            elif cross_dot[j,i,0]>0 or cross_dot[j,i,1]>0 or  cross_dot[j,i,2]>0:
                length2=distance_max[j,i]*distance_max[j,i]
            else:
                length2=0 
    #        print(length1,length2)    
    #        print(time.time()-time1)    
            
            length=np.sqrt(length1[j,i]*length1[j,i]+length2)
  
            if length<min_[j]:
                min_[j]=length
      
     return min_        
    


#    
#import time
#
##time1=time.time()
#number_of_triangle=1
##for i in range(100):
#
##triangle=np.random.rand(number_of_triangle,3,3)
#triangle=np.array([[[-1,-1,0],[1,-1,0],[0,1,0]]]).astype(float)
#point=np.array([-2,-2,-1])
#
#
##    print(length(triangle,point)) 
#
#print(length(number_of_triangle,triangle,point))

#    if length(number_of_triangle,triangle,point)<19 or length(number_of_triangle,triangle,point)>20:
#        break
#    print(triangle.dtype,point.dtype)   
#print(time.time()-time1) 
#print(time1)      
#       



#a=np.zeros(3)
#b=np.ones(3)
#c=np.zeros(3)
#sub(a,b,c)
#print(c)













#@nb.jit(nopython=True)
def length_to_point(number,triangle,point):
    min=1000000
    for i in range(number):
#        min_temp=np.dot(point-triangle[i,:],point-triangle[i,:])
#        print(np.dot(point-triangle[i,:],point-triangle[i,:]))
        min_temp=dot(point[0],triangle[i,0],point[1],triangle[i,1],point[2],triangle[i,2])
        if(min>min_temp):
            min=min_temp
    return min



@nb.vectorize(nopython=True)
def dot(a,b,c,d,e,f):
    return (a-b)*(a-b)+(c-d)*(c-d)+(e-f)*(e-f)


