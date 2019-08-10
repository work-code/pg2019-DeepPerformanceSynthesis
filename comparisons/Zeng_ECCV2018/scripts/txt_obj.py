import numpy as np
for model in range(9200,9400):
    print(model)
    a=np.loadtxt(r'C:\Users\Desktop\Project9-contrast\Project2\color_points\train_point_and_color{}_.txt'.format(model))
    filename=r'C:\Users\Desktop\Project9-contrast\Project2\color_points_obj\{}.obj'.format(model)
    with open(filename, 'w') as fh:
        for i in range(a.shape[0]):          
            fh.write("v {} {} {} {} {} {}\n".format(*np.array(a[i,0:3]),*np.array(a[i,3:6]/255)))
            
    