
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import isinsider
import numpy as np
import projection
import matplotlib.image as mpimg 
import gc

import random
import Net

net=Net.Net() 
camera=np.array([0,9,18,27])   
number_of_camera=camera.shape[0]

resolution_x=1024
resolution_y=1024
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
        R[1,1]=resolution_x*R[1,1]/200
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

def show_image(path,points):
    image=mpimg.imread(path)
    print(image.shape)
    plt.imshow(image)   
    plt.scatter(points[:,0], points[:,1], s=10, marker='.', c='r')       
    plt.pause(0.001)



optimizer = (optim.SGD(net.parameters(), lr=0.0001, momentum=0.9))
checkpoint = torch.load('1024_0_136.pt')
net.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_temp = checkpoint['epoch']
loss = checkpoint['loss']
number_sum=checkpoint['number_sum']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=net.to(device)
#net.train()
net.eval()
print(epoch_temp)   



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
step=0.00001
optimizer = optim.Adam(net.parameters(), lr=step)

for epoch in range(0,30):
    sum_loss=0
    for model in range(10,200):                                           
                for k in range(1):
                    print(r"D:\data\PVHM_0-9999\0-199\m_{}.obj".format(model))        
                    path =r"D:\data\PVHM_0-9999\0-199\m_{}.obj".format(model)
                    bounding_box=projection.bounding_box(path)
                    print(bounding_box)
                    time1=time.time()
                    
                   
                   
  
                    
                    number2=2
                    number1=100
                    
                    number=number1*number2
                     
                    points=np.zeros([number,3])
                    points[:,0]=np.random.uniform(bounding_box[0],bounding_box[1],number)
                    points[:,1]=np.random.uniform(bounding_box[2],bounding_box[3],number)
                    points[:,2]=np.random.uniform(bounding_box[4],bounding_box[5],number)

                    
                    points=torch.tensor(points)
                         
                    

                   
                    labels=np.zeros([number,2])  
                    for m in range(number2):
                        openEdges=isinsider.inside(path,number1,points[number1*m:number1*(m+1),:],labels[number1*m:number1*(m+1),:])
                    
                                        
                    if openEdges!=0:
                        break  
                    
                    value=isinsider.divide(points,labels)
                    points=np.array(value[0])
                    labels=np.array(value[1])
                    
                    
                    index = [i for i in range(points.shape[0])] 
                    random.shuffle(index)
                    points = points[index]
                    labels = labels[index]
            #        print(data,label)
                    
                    print(time.time()-time1)
    
                    number=points.shape[0]
                    print('点的数量{}'.format(number))
                    number_sum=number_sum+number
                    if number_sum>100000:
                        print(number_sum)
                        number_sum=0
                        step=0.7*step
                        optimizer = optim.Adam(net.parameters(), lr=step)
                    pamt=np.zeros([number_of_camera,3,4])
                    projection_points=np.zeros([number_of_camera,points.shape[0],2])
                    
                    pamt=read_cal(36)
                    
                    new_points=np.copy(points)
                    points[:,1]=-new_points[:,2]
                    points[:,2]=new_points[:,1]
                    
                    for i in range(number_of_camera):
                        print(camera[i])
                         
                        projection_points[i,:,:]=projection.projection(points,pamt[camera[i],:,:],number)
                        
                    photo=torch.empty(number_of_camera,3,resolution_x,resolution_y)
                    for i in range(number_of_camera):                    
                        photo[i,:]=torch.tensor(np.transpose(plt.imread(r"D:\data\PVHM_0-9999\rendered_images\{}_{}_{}x{}.png".format(model,camera[i],resolution_x,resolution_y))).astype(float))[0:3]
                        show_image(r"D:\data\PVHM_0-9999\rendered_images\{}_{}_{}x{}.png".format(model,camera[i],resolution_x,resolution_y),projection_points[i,:,:])
                    
                    ticks1 = time.time()
                    running_loss = 0.0
                    photo=photo.to(device)
    
                   
                      
                    inputs=torch.tensor(projection_points)
    
                    inputs = inputs.to(device)
                    label=torch.tensor(labels).cuda().type(torch.float)
                    lable=label.to(device)
        
                    net=net.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    time1=time.time()
                    outputs = net(photo,inputs).type(torch.float)
                    print(time.time()-time1)
                    loss=torch.nn.functional.binary_cross_entropy(outputs,label)
                    print(loss)
    
                    loss.backward()
                    optimizer.step()
     
                    running_loss += loss.item()
                    print('loss:',loss)   
                    sum_loss=sum_loss+loss.item()
                    np.savetxt(str(epoch)+'_'+ str(model)+'.txt',np.array([running_loss/number]))
                        
    
                    running_loss = 0.0
                    print(time.time()-ticks1)
                    
                    torch.save({
                    'epoch': epoch,
                    'model':model,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,  
                    'number_sum':number_sum,
                    'step':step,
                    },"./model/{}_{}_{}.pt".format(resolution_x,epoch,model))
                    torch.cuda.empty_cache()
                    gc.collect()
    np.savetxt('sum_loss_{}.txt'.format(epoch),np.array([sum_loss]))                
print('Finished Training')
