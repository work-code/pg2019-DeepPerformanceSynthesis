import numpy
import vtk
import numba as nb
@nb.jit
def projection(points,pamt,number):
    point=numpy.ones([1,number])
#    print(point)
    point=numpy.vstack((numpy.transpose(points),point))
#    print(point,pamt)
    point=numpy.dot(pamt,point) 
#    print(point.shape)
    point=numpy.transpose(point)
#    print(point)
    for i in range(number):
        if point[i,2]!=0:
            point[i,:]=point[i,:]/point[i,2]  
#    print(point)            
    return point[:,0:2]


def bounding_box(path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)    
    reader.Update()
    pdata = reader.GetOutput()
    return pdata.GetBounds()
   

