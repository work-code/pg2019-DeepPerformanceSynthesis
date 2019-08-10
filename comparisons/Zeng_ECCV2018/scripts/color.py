import projection
import numpy as np
import qr
import math
import vtk
import torch
import matplotlib.pyplot as plt
import time
import numba as nb

@nb.jit(nopython=True)
def color_points_color(pamt,photo,points,projection_points,color,weight):
    for i in range(points.shape[0]):
        for j in range(pamt.shape[0]):   
            projection_x=projection_points[j,i,0]
            projection_y=projection_points[j,i,1]
            x_ceil=math.ceil(projection_x)
            x_floor=x_ceil-1
            y_ceil=math.ceil(projection_y)
            y_floor=y_ceil-1 

            y=(photo[j,:,x_ceil,y_ceil]*(x_floor-projection_x)*(y_floor-projection_y)
                    +photo[j,:,x_floor,y_ceil]*(projection_x-x_ceil)*(y_floor-projection_y)
                    +photo[j,:,x_ceil,y_floor]*(x_floor-projection_x)*(projection_y-y_ceil)
                    +photo[j,:,x_floor,y_floor]*(projection_x-x_ceil)*(projection_y-y_ceil))      

            color[i,:]=color[i,:]+weight[i,j]*y*255
            
@nb.jit(nopython=True)
def wight_points(points,normals,pamt,direction,weight):
  for i in range(points.shape[0]): 
     sum=0
#         print(i)
#             print(normals[i,:],direction[i,j,:])
     if np.dot(normals[i,:],normals[i,:])==0:
#                 print(111111)
#                 weight[i,j]=0
         continue 
     else:
         for j in range(pamt.shape[0]):  
#                 print(1111)     
             weight[i,j]=np.dot(normals[i,:],direction[i,j,:])/np.sqrt(np.dot(normals[i,:],normals[i,:]))/np.sqrt(np.dot(direction[i,j,:],direction[i,j,:]))
             if weight[i,j]<0:
                 weight[i,j]=0
             sum=sum+weight[i,j]               
         for j in range(pamt.shape[0]):
#                print(222)  
            if sum>0:
                weight[i,j]=weight[i,j]/sum
            else :
                weight[i,j]=0  
#             print(pamt.shape[0]) 
#             print(333)        
#             point_list.append(points[i,:])
#             projection_points_list.append(projection_points[:,i,:])
#             weight_list.append(weight[i,:])
#             print(444444)  
  
    
@nb.jit
def color_points(photo,points,pamt,normals,model):
#    np.save('points',points)
#    print(photo[0,0,800,600]) 
#    points_temp=points.copy()
    site=np.loadtxt('camera_site.txt')
  
    print(pamt.shape[0])
    direction=np.zeros((points.shape[0],pamt.shape[0],3))
    time1=time.time()
    for i in range(points.shape[0]):
        for j in range(pamt.shape[0]):
            direction[i,j,:]=site[j,:]-points[i,:]
    projection_points=np.zeros((pamt.shape[0],points.shape[0],2))
#    print(pamt.shape[0])
    
    for i in range(pamt.shape[0]):     
        projection_points[i,:,:]=projection.projection(points,pamt[i,:,:],points.shape[0])
    weight=np.zeros((points.shape[0],pamt.shape[0])) 

#    point_list=[]
#    projection_points_list=[]
#    weight_list=[]
    time1=time.time()
  
    wight_points(points,normals,pamt,direction,weight)
    
    
    print(12234235234243,time.time()-time1)                
    print(points.shape)        
#    weight=np.array(weight_list)
#    projection_points=np.array(projection_points_list).transpose(1,0,2)    
#    points=np.array(point_list)      
    print(points.shape,projection_points.shape,weight.shape)
#    print(len(point_list))
    color=np.zeros((points.shape[0],3))          
    color_points_color(pamt,photo,points,projection_points,color,weight)         
    point_and_color=np.zeros((points.shape[0],6))    
    point_and_color[:,0:3]=points
    point_and_color[:,3:6]=color.astype(int)
    np.savetxt('./color_points/train_point_and_color'+str(model)+"_"+".txt",point_and_color)          
    return color   
 


def camera_site(pamt):          
    number_of_camera=pamt.shape[0]
    site=np.zeros([number_of_camera,3])
    for i in range(number_of_camera):         
        (R,Q)=qr.RQ(pamt[i,:,0:3].copy())
        T=np.dot(np.linalg.inv(R),pamt[i,:,:])

        site[i,:]=-np.dot(Q.transpose(),T[:,3])
    np.savetxt('camera_site.txt',site) 
    return site 


def vtk_read(path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    
    reader.Update()
    pdata = reader.GetOutput()
    
    # check if the stl file is closed
    featureEdge = vtk.vtkFeatureEdges()
    featureEdge.FeatureEdgesOff()
    featureEdge.BoundaryEdgesOn()
    featureEdge.NonManifoldEdgesOn()
    featureEdge.SetInputData(pdata)
    featureEdge.Update()

    

    # pass pdata through a triangle filter
    tr= vtk.vtkTriangleFilter()
    tr.SetInputData(pdata)
    tr.PassVertsOff()
    tr.PassLinesOff()
    tr.Update()
    
    # normals filter
    pnormal = vtk.vtkPolyDataNormals()
    pnormal.SetInputData(tr.GetOutput())
    pnormal.AutoOrientNormalsOff()
    pnormal.ComputePointNormalsOn()
    pnormal.ComputeCellNormalsOff() 
    pnormal.SplittingOff()
    pnormal.ConsistencyOn()
    pnormal.FlipNormalsOn()
    pnormal.Update()
    pdata = pnormal.GetOutput()
    
    # create a vtkSelectEnclosedPoints filter
    filter = vtk.vtkSelectEnclosedPoints()
    filter.SetSurfaceData(pdata)
    print(pdata.GetNumberOfPoints()) 
    print(pdata.GetPointData().GetNumberOfTuples())
#    print(pdata)
    obj_points=np.zeros([pdata.GetNumberOfPoints(),3])
    obj_normals=np.zeros([pdata.GetNumberOfPoints(),3])
    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(pdata)
    
    for i in range(pdata.GetNumberOfPoints()): 
        obj_points[i,:]=pdata.GetPoint(i)  
        obj_normals[i,:]=pdata.GetPointData().GetNormals().GetTuple(i)
    return (obj_points,obj_normals)    
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
    
    site=np.zeros([4,3])  
    for i in range(4):
        site[i,:]=-np.dot(np.array(T_list[9*i])[0:3,0:3].transpose(),np.array(T_list[9*i])[0:3,3])
    np.savetxt('camera_site.txt',site) 
    
    for i in range(camera_number):
        pamt_list[i]=np.dot(np.array(R_list[i]),T_list[i])
        
    return pamt_list

start=9200
resolution_x=1024
resolution_y=1024
camera=np.array([0,9,18,27])  
number_of_camera=len(camera)
pamt_list=read_cal(36)
pamt=np.zeros([number_of_camera,3,4])
for i in range(number_of_camera):
    pamt[i,:,:] = pamt_list[camera[i]]
#site=camera_site(pamt.copy())







for model in range(0,200): 
            time1=time.time()

            path=r"./obj/sphere_{}.obj".format(start+model)
            print(path)
            
            (obj_points,obj_normals)=vtk_read(path)    


            
            photo=np.array(torch.empty(number_of_camera,3,resolution_x,resolution_y))
            for i in range(number_of_camera):                    
                photo[i,:]=np.transpose(plt.imread(r"D:\data\PVHM_0-9999\rendered_images\{}_{}_{}x{}.png".format(model+start,camera[i],resolution_x,resolution_y))).astype(float)[0:3]
                plt.imshow(np.transpose(photo[i,:]))
                plt.show()
            
            print(np.max(obj_normals))
#            print(obj_normals[0:100])
            max=-1000
            index=-1000
            for i in range(obj_points.shape[0]):
                if obj_points[i,1]>max:
                    index=i
                    max=obj_points[i,1]
            print(max,index)        
            if index!=-1000 and obj_normals[index,1]<0:
                obj_normals=-np.array(obj_normals)  
            
            bouding_box=projection.bounding_box(path)
            print(bouding_box)               
                
          
            np.savetxt('points_'+str(model+start)+"_"+".txt",obj_points)         
            color_points(photo,obj_points,pamt,obj_normals,model+start)
            print(123123,time.time()-time1)
# In[]
import vtk
import numpy as np
import numpy
            
def vtk_read(path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    
    reader.Update()
    pdata = reader.GetOutput()
    
    # check if the stl file is closed
    featureEdge = vtk.vtkFeatureEdges()
    featureEdge.FeatureEdgesOff()
    featureEdge.BoundaryEdgesOn()
    featureEdge.NonManifoldEdgesOn()
    featureEdge.SetInputData(pdata)
    featureEdge.Update()

    

    # pass pdata through a triangle filter
    tr= vtk.vtkTriangleFilter()
    tr.SetInputData(pdata)
    tr.PassVertsOff()
    tr.PassLinesOff()
    tr.Update()
    
    # normals filter
    pnormal = vtk.vtkPolyDataNormals()
    pnormal.SetInputData(tr.GetOutput())
    pnormal.AutoOrientNormalsOff()
    pnormal.ComputePointNormalsOn()
    pnormal.ComputeCellNormalsOff() 
    pnormal.SplittingOff()
    pnormal.ConsistencyOn()
    pnormal.FlipNormalsOn()
    pnormal.Update()
    pdata = pnormal.GetOutput()
    
    # create a vtkSelectEnclosedPoints filter
    filter = vtk.vtkSelectEnclosedPoints()
    filter.SetSurfaceData(pdata)
    print(pdata.GetNumberOfPoints()) 
    print(pdata.GetPointData().GetNumberOfTuples())
#    print(pdata)
    obj_points=np.zeros([pdata.GetNumberOfPoints(),3])
    obj_normals=np.zeros([pdata.GetNumberOfPoints(),3])
    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(pdata)
    
    for i in range(pdata.GetNumberOfPoints()): 
        obj_points[i,:]=pdata.GetPoint(i)  
        obj_normals[i,:]=pdata.GetPointData().GetNormals().GetTuple(i)
        
        
    obj_polygons=numpy.zeros([pdata.GetNumberOfCells(),3]).astype(np.int)
    for i in range(pdata.GetNumberOfCells()): 
        cell=vtk.vtkIdList()
        pdata.GetCellPoints(i,cell)
        for j in range(3):
            obj_polygons[i,j]=cell.GetId(j) 
#            print(cell.GetId(j) )
    return (obj_points,obj_normals,obj_polygons)    


def export_obj(vertices, triangles, filename):
    """
    Exports a mesh in the (.obj) format.
    """
    
    with open(filename, 'w') as fh:
        
        for v in vertices:
            fh.write("v {} {} {} 1 0 0\n".format(*v))
            
        for f in triangles:
            fh.write("f {} {} {}\n".format(*(f + 1)))









