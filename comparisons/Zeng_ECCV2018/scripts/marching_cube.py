# -*- coding: utf-8 -*-


import isinsider
import projection
import numpy as np
import mcubes
import time
import numba as nb
from Net import Net
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


@nb.jit(nopython=True)
def Position(probability):
    a=probability[0]*(1-probability[1])
    b=probability[0]*probability[1]
    c=probability[1]*(1-probability[0])
    if b>=a and b>=c:
        return np.array([1,1])
    elif c>=a and c>=b:
        return np.array([0,1])
    else:    
        return np.array([1,0])

#marching cube
def marching_cube(resolution,bounding_box,model_no):
    u= np.load('leabels.npy')
    print(u.shape)
    
      # Extract the 0-isosurface
    vertices, triangles = mcubes.marching_cubes(u, 0.1)
    print(np.max(vertices[:,2])) 

    print(bounding_box,resolution)
    for i in range(vertices.shape[0]):
        for j in range(3):
            vertices[i,j]=vertices[i,j]/resolution*(bounding_box[2*j+1]-bounding_box[2*j])+bounding_box[2*j]
    print(np.max(vertices[:,1]))        
 #   mcubes.export_mesh(vertices, triangles, "./dae/sphere_"+str(model_no)+".dae", "MySphere")
    mcubes.export_obj(vertices, triangles, "./obj/sphere_"+str(model_no)+".obj")
    return vertices




 
#def accurate_model(bounding_box,number,path):
#    points=np.zeros([number*number*number,3])
#    labels=np.zeros([number*number*number,2])
##    time1=time.time()
#   set_points(bounding_box,points)          
##    print(time.time()-time1)          
#
#                 
##    print(1111)
##    time1=time.time()    
#    outputs=isinsider.inside(path,number*number*number,points,labels)
##    print(time.time()-time1)
#    return labels


@nb.jit(nopython=True) 
def set_points(bounding_box,number,points):
#    print(bounding_box)
    for i in range(number):
        for j in range(number):
            for k in range(number):
                points[number*number*i+number*j+k,:]=np.array([(bounding_box[1]-bounding_box[0])*i/(number-1)+bounding_box[0],(bounding_box[3]-bounding_box[2])*j/(number-1)+bounding_box[2],(bounding_box[5]-bounding_box[4])*k/(number-1)+bounding_box[4]])
    return points


def prediction_model(number_of_camera,number,net,bounding_box):
#    time1=time.time()
    points=np.zeros([number*number*number,3])
    set_points(bounding_box,number,points)
    projection_points=np.zeros([number_of_camera,number*number*number,2])
    pamt=read_cal(36)
    
    for i in range(number_of_camera):
       
        projection_points[i,:,:]=projection.projection(points,pamt[camera[i],:,:],number*number*number)    
                  
    with torch.no_grad():
        inputs=torch.tensor(projection_points)
        inputs=inputs.to(device)  
#        print(time.time()-time1)       
        outputs = net(photo,inputs)
        print(1234242434,torch.cuda.max_memory_allocated())
    return (outputs,points)   

  
@nb.jit(nopython=True)
def  prediction_model_label(number,outputs,single_label,labels):
    for i in range(number):
        for j in range(number):
            for k in range(number):                            
            
                a=Position(outputs[i*number*number+j*number+k,:])
                labels[i*number*number+j*number+k,:]=a               
            
                if a[0]==1 and a[1]==1:
                    single_label[i,j,k]=0
                elif a[0]==1 and a[1]==0:    
                    single_label[i,j,k]=-1
                else:
                    single_label[i,j,k]=1
                    

@nb.jit(nopython=True)                    
def prediction_model_label_same(number,single_label,label_number):
    for i in range(number):
        for j in range(number):
            for k in range(number):                                           
                single_label[i,j,k]=label_number
              
                 
def read_cal(camera_number):    
    intrinsic_path=r'D:\data\PVHM_Tools\flow_renderer\intrinsic.txt'
    extrinsic_path=r'D:\data\PVHM_Tools\flow_renderer\extrinsic.txt'
    R_list=[]
    T_list=[]
    pamt_list=np.zeros([camera_number,3,4])
    f=open(intrinsic_path)
    intrinsic_lines = f.readlines()  
    g=open(extrinsic_path) 
    extrinsic_lines = g.readlines()  

    for i in range(18):
        R=np.zeros([3,4])
        for j in range(3):
            R[j,0:3]=list(map(float, intrinsic_lines[3*i+j].split()))    
        R[0,0]=resolution_x*R[0,0]/200
        R[1,1]=resolution_x*R[1,1]/200
        R[0,2]=(resolution_x+1.0)/2
        R[1,2]=(resolution_y+1.0)/2
        R_list.append(R)
        
        T=np.zeros([4,4])
        for j in range(3):
            T[j,:]=list(map(float, extrinsic_lines[3*i+j].split()))
        T[3,3]=1
        T_list.append(T)
        
    for i in range(18):
        R=np.zeros([3,4])
        for j in range(3):
            R[j,0:3]=list(map(float, intrinsic_lines[3*i+j].split()))    
        
        R[0,0]=resolution_x*R[0,0]/200
        R[1,1]=resolution_y*R[1,1]/200
        R[0,2]=(resolution_x+1.0)/2
        R[1,2]=(resolution_y+1.0)/2
        
        R_list.append(R)
        
        T=np.zeros([4,4])
        for j in range(3):
            T[j,:]=list(map(float, extrinsic_lines[3*i+j].split()))
        T[0,0:3]=-T[0,0:3]
        T[2,0:3]=-T[2,0:3]    
        T[3,3]=1
        T_list.append(T)   
    

   
        
    
    for i in range(camera_number):
        pamt_list[i]=np.dot(np.array(R_list[i]),T_list[i])
        
    return pamt_list
               

net=Net()
optimizer = (optim.SGD(net.parameters(), lr=0.001, momentum=0.9))
checkpoint = torch.load('4_7_18_1_5.pt')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_temp = checkpoint['epoch']
loss = checkpoint['loss']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=net.to(device)

resolution_x=1024
resolution_y=1024

start=9200
camera=np.array([0,9,18,27])   
number_of_camera=len(camera)
for model in range(0,200):
    time9=time.time()
    path=r"D:\data\PVHM_0-9999\{}-{}\m_{}.obj".format(start,start+199,model)
    bounding_box=np.array(projection.bounding_box(path))
    big_box=np.zeros(6)
    boundingbox=np.copy(bounding_box)   
    bounding_box=np.array([boundingbox[0],boundingbox[1],-boundingbox[5],-boundingbox[4],boundingbox[2],boundingbox[3]])
    
           
    big_box[0]=(bounding_box[0]*2.2-bounding_box[1]*0.2)/2
    big_box[1]=(bounding_box[1]*2.2-bounding_box[0]*0.2)/2   
    
    big_box[2]=(bounding_box[2]*2.5-bounding_box[3]*0.5)/2
    big_box[3]=(bounding_box[3]*2.5-bounding_box[2]*0.5)/2  
    
    big_box[4]=(bounding_box[4]*2.1-bounding_box[5]*0.1)/2
    big_box[5]=(bounding_box[5]*2.1-bounding_box[4]*0.1)/2
    
    
    bounding_box=big_box 
    
    print(bounding_box)    
        
    number1=40
    number2=5
    number=number1*number2
    
    photo=torch.empty(number_of_camera,3,resolution_x,resolution_y)        
    for i in range(number_of_camera):    
        photo[i,:]=torch.tensor(np.transpose(plt.imread(r"D:\data\PVHM_0-9999\rendered_images\{}_{}_{}x{}.png".format(model+start,camera[i],resolution_x,resolution_y))).astype(float))[0:3] 
 #       print("C:/Users/Desktop/Project2/data/RENDER/anim_"+str(number_fo_anim)+"/model_"+str(model)+"_anim_"+str(number_fo_anim)+"/cameras_cam0"+str(camera[i])+"/alpha_00"+str(number_of_photo)+"1.png") 
    photo=photo.to(device)
    
               
              
    
        
    
     
    
    time2=time.time()
    single_label_mesh= np.zeros([number1,number1,number1])   
    single_label_center= np.zeros([number1-1,number1-1,number1-1])  
    labels_mesh= np.zeros([number1*number1*number1,2])   
    labels_center= np.zeros([number1*number1*number1,2])   
    single_label=np.ones([number,number,number]) 
       
    
    (outputs_center,points)=prediction_model(number_of_camera,number1-1,net,bounding_box/number1*(number-1)) 
    
    outputs_center=outputs_center.cpu()
    outputs_center=np.array(outputs_center)
    time1=time.time()
    (outputs_mesh,grid_points)=prediction_model(number_of_camera,number1,net,bounding_box) 
    print(time.time()-time1)
    outputs_mesh=outputs_mesh.cpu()
    outputs_mesh=np.array(outputs_mesh)
    
    prediction_model_label(number1,outputs_mesh,single_label_mesh,labels_mesh)
    prediction_model_label(number1-1,outputs_center,single_label_center,labels_center)
    print(grid_points.shape)
   
    boundary_list=[]
    range_list=[]
    print(time.time()-time2)
    time2=time.time()
    times=0
    for i in range(number1-1):
        for j in range(number1-1):
            for k in range(number1-1):
                sum_inside=0
                sum_outside=0
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            sum_inside=sum_inside+labels_mesh[(i+ii)*number1*number1+(j+jj)*number1+k+kk,0]
                            sum_outside=sum_outside+labels_mesh[(i+ii)*number1*number1+(j+jj)*number1+k+kk,1]
    #            print(sum_outside,sum_inside) 
                if  single_label_center[i,j,k]==0: 
                    times=times+1
                    boundary=np.zeros(6)
    #                print(grid_points.shape)
                    for m in range(3):                    
                        boundary[2*m]=grid_points[(i)*number1*number1+(j)*number1+k,m]
                        boundary[2*m+1]=grid_points[(i+1)*number1*number1+(j+1)*number1+k+1,m]
                    boundary_list.append(boundary)    
                    range_list.append([i,j,k])
                    
                    
                   
                    continue
                elif sum_outside==0 and sum_inside==8:
                    prediction_model_label_same(number2,single_label[number2*i:number2*(i+1),number2*j:number2*(j+1),number2*k:number2*(k+1)],-1)
                    continue
                elif sum_inside==0 and sum_outside==8:
                    prediction_model_label_same(number2,single_label[number2*i:number2*(i+1),number2*j:number2*(j+1),number2*k:number2*(k+1)],1)
                    continue
                else :
                    times=times+1
                    boundary=np.zeros(6)
                    for m in range(3):                    
                        boundary[2*m]=grid_points[(i)*number1*number1+(j)*number1+k,m]
                        boundary[2*m+1]=grid_points[(i+1)*number1*number1+(j+1)*number1+k+1,m]
                    boundary_list.append(boundary)      
                    range_list.append([i,j,k])
                                           
                    
                    continue
    print(time.time()-time2)   
    time2=time.time()                 
    batch_size=9*9*9
    number_of_batch=int((len(boundary_list)-1)/batch_size)+1
    print(number_of_batch,number1,number2)
    for batch in range(number_of_batch):
        if batch!=number_of_batch-1:
            size=batch_size
        else:
            size=len(boundary_list)-batch_size*(number_of_batch-1)
        points=np.zeros([number2*number2*number2*size,3])
        for i in range(size):
            set_points(boundary_list[batch*batch_size+i],number2,points[i*number2*number2*number2:(i+1)*number2*number2*number2,:])
            
        projection_points=np.zeros([number_of_camera,number2*number2*number2*size,2])
        pamt=read_cal(36)
    
        for i in range(number_of_camera):           
            projection_points[i,:,:]=projection.projection(points,pamt[camera[i],:,:],number2*number2*number2*size)    
                  
        with torch.no_grad():
            inputs=torch.tensor(projection_points)
            inputs=inputs.to(device)        
            output = net(photo,inputs)
        output=np.array(output.cpu())

        labels_temp= np.zeros([number2*number2*number2,2])
        for m in range(size):
            i=range_list[batch*batch_size+m][0]
            j=range_list[batch*batch_size+m][1]
            k=range_list[batch*batch_size+m][2]
    
            prediction_model_label(number2,output[m*number2*number2*number2:(m+1)*number2*number2*number2,:],single_label[number2*i:number2*(i+1),number2*j:number2*(j+1),number2*k:number2*(k+1)],labels_temp)             
           
    print(time.time()-time2,times)
                 
    np.save('leabels',single_label)   
        
    vertices=marching_cube(number,bounding_box,model+start)       
    print(time.time()-time9)  
             

  




