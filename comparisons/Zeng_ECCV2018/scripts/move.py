import os,shutil

start=7200
resoultion=512
fpath=r'C:\Users\Desktop\Project9-contrast\image_renderer\output_image_{}\{}-{}'.format(resoultion,start,start+199) 
if not os.path.exists(fpath):
        os.makedirs(fpath)                
for model in range(200):
    

    for i in range(20):
        srcfile=r'C:\Users\Desktop\Project9-contrast\image_renderer\output_image_{}\{}_{}.png'.format(resoultion,model+start,i)
     
        path=r'C:\Users\Desktop\Project9-contrast\image_renderer\output_image_{}\{}-{}\{}'.format(resoultion,start,start+199,model+start)
        if not os.path.exists(path):
            os.makedirs(path) 
#        print(srcfile)
        shutil.move(srcfile,path)         


# In[]
import os,shutil 

start=4600
resolution=200


fpath=r'D:\data\PVHM_0-9999\rendered_images\{}x{}'.format(resolution,resolution) 
if not os.path.exists(fpath):
        os.makedirs(fpath)                
for model in range(200):
    

    for i in range(0,36,9):
        srcfile=r'D:\data\PVHM_0-9999\rendered_images\{}_{}_{}x{}.png'.format(model+start,i,resolution,resolution)
     
        path=r'D:\data\PVHM_0-9999\rendered_images\{}x{}\\'.format(resolution,resolution)
        if not os.path.exists(path):
            os.makedirs(path) 
#        print(srcfile)
        shutil.move(srcfile,path)          
 