import numpy as np
import projection
start=1400
for model in range(67,68):
    print(model)
    obj_path=r'D:\data\PVHM_0-9999\{}-{}\m_{}.obj'.format(start,start+199,model)
    bounding_box=np.array(projection.bounding_box(obj_path))  
    big_box=np.zeros(6)
    boundingbox=np.copy(bounding_box)   
    bounding_box=np.array([boundingbox[0],boundingbox[1],-boundingbox[5],-boundingbox[4],boundingbox[2],boundingbox[3]])
    
           
    big_box[0]=(bounding_box[0]*2.3-bounding_box[1]*0.3)/2
    big_box[1]=(bounding_box[1]*2.3-bounding_box[0]*0.3)/2   
    
    big_box[2]=(bounding_box[2]*2.1-bounding_box[3]*0.1)/2
    big_box[3]=(bounding_box[3]*2.1-bounding_box[2]*0.1)/2  
    
    big_box[4]=(bounding_box[4]*2.1-bounding_box[5]*0.1)/2
    big_box[5]=(bounding_box[5]*2.1-bounding_box[4]*0.1)/2
    
    txt_path=r'C:\Users\Desktop\Project9-contrast\Project2\color_points\train_point_and_color{}_.txt'.format(model+start)
    print(txt_path)
    point=np.loadtxt(txt_path)
    point_number=0
    
    label=(point[:,0]<big_box[1])*(point[:,0]>big_box[0])*(point[:,1]<big_box[3])*(point[:,1]>big_box[2])*(point[:,2]<big_box[5])*(point[:,2]>big_box[4])
    number=np.sum(label)
    print(number)
    new_point=np.zeros([number,6])
    print(point.shape)
    if number==point.shape[0]:
        new_point=point
    else :
        print(1111111)
        k=0
        for i in range(point.shape[0]):
            if label[i]==1:
           
                new_point[k]=point[i]
                k=k+1
        print(k)        
        if k!=number:
            print('warning')
    np.savetxt(r'C:\Users\Desktop\Project9-contrast\Project2\new_color_points\train_point_and_color{}_.txt'.format(model+start),new_point)       
# In[]



txt_path=r'C:\Users\Desktop\Project9-contrast\Project2\new_color_points\train_point_and_color{}_.txt'.format(1467)
print(txt_path)
point=np.loadtxt(txt_path)  
print(point.shape)  