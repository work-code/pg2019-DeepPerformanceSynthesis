# Script to check if a point lies inside or outside a closed STL file
#import sys
import vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import length
import numpy 
import torch
import numpy as np



#if locate on the surface of the model

def label_surface(obj_polygons,pdata_GetNumberOfCells,distance,number,labels,points):
    obj_polygons_temp=torch.tensor(obj_polygons.copy())    
    distance=length.length(pdata_GetNumberOfCells,obj_polygons_temp,points)    
    for i in range(number):
        if distance[i]<0.0001:
            labels[i,0]=labels[i,1]=1

    
            
#create a points polydata
def getPolydata(i,j,k):
    
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(i,j,k) # selected a point here which lies inside the stl surface , Bounds of the stl file are (-0.5, -0.5, -0.5, 0.5, 0.5, 0)
    pts_pd = vtk.vtkPolyData()
    pts_pd.SetPoints(pts)
    return pts_pd




#Used to distinguish the location of the point 
def inside(path,number,points,labels):
#    time1=time.time()
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
    openEdges = featureEdge.GetOutput().GetNumberOfCells()
    
    if openEdges != 0:
        print("STL file is not closed")
        print(openEdges)
 #       return openEdges

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

    # checking for consistency of IsInside method
 
    filter.SetTolerance(0.00001)
    
    time1=time.time()  
    for i in range(number):  
        filter.SetInputData(getPolydata(points[i,0],points[i,1],points[i,2]))
        filter.Update()            
        labels[i,0]=filter.IsInside(0)
        labels[i,1]=1-labels[i,0] 
    distance=numpy.zeros(number)   
    
    obj_polygons=numpy.zeros([pdata.GetNumberOfCells(),3,3])
    for i in range(pdata.GetNumberOfCells()): 
        cell=vtk.vtkIdList()
        pdata.GetCellPoints(i,cell)
        for j in range(3):
            obj_polygons[i,j,:]=pdata.GetPoint(cell.GetId(j))
             
    pdata_GetNumberOfCells=pdata.GetNumberOfCells()
    label_surface(obj_polygons,pdata_GetNumberOfCells,distance,number,labels,points) 
                               
    print(time.time()-time1) 
    return 0




def divide(points,labels):
    number_in=0
    number_out=0
    number_surface=0
    for i in range(points.shape[0]):
        if labels[i,0]==1 and labels[i,1]==0:
            number_in+=1
        elif labels[i,1]==1 and labels[i,0]==0:
            number_out+=1
        else:
            number_surface+=1
        
    if  number_in>number_out:  
        print('number_in>number_out')
        divide_point=numpy.zeros([2*number_out+number_surface,3])
        divide_labels=numpy.zeros([2*number_out+number_surface,2])
        j=0
        for i in range(points.shape[0]):
            if labels[i,1]==1 and labels[i,0]==0:
                divide_point[2*j+1,:]=points[i]
                divide_labels[2*j+1,:]=labels[i]
                j=j+1
        j=0        
        for i in range(points.shape[0]):
            
            if labels[i,0]==1 and labels[i,1]==0:
                divide_point[2*j,:]=points[i]
                divide_labels[2*j,:]=labels[i]
                j=j+1
                if j==number_out:
                    k=0
                    for i in range(points.shape[0]):                                                
                        if labels[i,0]==1 and labels[i,1]==1:
                            divide_labels[2*number_out+k,:]=labels[i]
                            divide_point[2*number_out+k,:]=points[i]
                            k=k+1
                    value=[]
                    value.append(divide_point)
                    value.append(divide_labels)
                    return value  
    else:
       
        divide_point=numpy.zeros([2*number_in+number_surface,3])
        divide_labels=numpy.zeros([2*number_in+number_surface,2])
        j=0
        for i in range(points.shape[0]):
            if labels[i,0]==1 and labels[i,1]==0:
                divide_point[2*j+0,:]=points[i]
                divide_labels[2*j+0,:]=labels[i]
                j=j+1
        j=0        
        for i in range(points.shape[0]):
            if labels[i,1]==1 and labels[i,0]==0:
                divide_point[2*j+1,:]=points[i]
                divide_labels[2*j+1,:]=labels[i]
#                print(divide_labels[2*j+1,:])
                j=j+1
                if j==number_in:
                    k=0
                    for i in range(points.shape[0]):                        
                        if labels[i,0]==1 and labels[i,1]==1:
                            divide_labels[2*number_in+k,:]=labels[i]
                            divide_point[2*number_in+k,:]=points[i]
                            k=k+1      
                    value=[]
                    value.append(divide_point)
                    value.append(divide_labels)
                    return value 




for model in range(1):
    path =r"D:\data\PVHM_0-9999\200-399\m_{}.obj".format(model)
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
    openEdges = featureEdge.GetOutput().GetNumberOfCells()
    
    if openEdges != 0:
        print("STL file is not closed")
        print(openEdges)

