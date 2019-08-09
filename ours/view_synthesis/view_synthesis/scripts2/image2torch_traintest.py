"""
A lot of the following code is a rewrite of:	
https://github.com/deepmind/gqn-datasets/data_reader.py	
"""
import sys
import time
import os
import collections
import torch
import numpy
import scipy.io as scio

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['base_path',  'scene_size_train', 'scene_size_test','frame_size', 'sequence_size']
)
Scene = collections.namedtuple('Scene', ['low_images', 'super_images'])

_DATASETS = dict(
    fruits=DatasetInfo(
        base_path='G:/2-paper/ResearchWork5/ResearchWork5_zz_finalversion/code/ours/view_synthesis/view_synthesis/data/' + 'PVHM',
        scene_size_train=[1,8000],
        scene_size_test=[8001,12000],
        frame_size=128,
        sequence_size=1),
)

model = 'model_28400'

def _get_dataset_files(dataset_info, mode, root):
    """Generates lists of files for a given dataset version."""
    basepath = dataset_info.basepath
    base = os.path.join(root, basepath, mode)
    if mode == 'train':
        num_files = dataset_info.train_size
    else:
        num_files = dataset_info.test_size

    files = sorted(os.listdir(base))

    return [os.path.join(base, file) for file in files]


def encapsulate(frames, cameras):
    return Scene(low_images=low_images, super_images = super_images)

def convert_raw_to_numpy(dataset_info, low_images, super_images, pt_dir):
    scene = encapsulate(low_images, super_images,)
    with open(pt_dir, 'wb') as f:
        torch.save(scene, f)


def show_frame(frames, scene, views):
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.imshow(frames[scene,views])
    plt.show()

def read_images(image_dir):
    raw_images = []
    with open(image_dir,'rb') as f:
        jpeg_data = f.read()
        raw_images.append(jpeg_data)
    raw_images = numpy.array(raw_images)
    raw_images = raw_images.astype('object')    
    return raw_images


if __name__ == '__main__':

    DATASET = 'fruits'
    dataset_info = _DATASETS[DATASET]

    torch_dataset_path_bias = dataset_info.base_path + '/torch_super/'
    if os.path.exists(torch_dataset_path_bias) == False:
        os.mkdir(torch_dataset_path_bias)
    
    torch_dataset_path_bias = torch_dataset_path_bias +'/' + model +'/'
    if os.path.exists(torch_dataset_path_bias) == False:
        os.mkdir(torch_dataset_path_bias)
        
    torch_dataset_path_bias = torch_dataset_path_bias + '/train/'
    if os.path.exists(torch_dataset_path_bias) == False:
        os.mkdir(torch_dataset_path_bias)
        
    torch_dataset_path_bias = torch_dataset_path_bias + '/bias_0/'
    if os.path.exists(torch_dataset_path_bias) == False:
        os.mkdir(torch_dataset_path_bias)
    
    file_names_bias = dataset_info.base_path + '/val/' + model + '/bias_0/'
    tot = 0
    for g in range(2,5+1):
        observation = g
        file_names = file_names_bias + 'observation_' + str(g) + '/'
        
        torch_dataset_path = torch_dataset_path_bias
            
        for l in range(dataset_info.scene_size_train[0],dataset_info.scene_size_train[1]+1):

            pt_dir = os.path.join(torch_dataset_path, f'{tot}.pt')

            
            image_dir = file_names + '/Scene' + str(l) + '/' + 'query.png'
            low_images = read_images(image_dir)
            
            image_dir = file_names + '/Scene' + str(l) + '/' + 'superimage.png'
            super_images = read_images(image_dir)
            
            #print(f' [-] converting scene {l} into {pt_dir}')
            convert_raw_to_numpy(dataset_info, low_images, super_images, pt_dir)
            print(f' [-] converting scene {l} into {pt_dir}')
            
            tot = tot+1
        
    torch_dataset_path_bias = dataset_info.base_path + '/torch_super/'
    if os.path.exists(torch_dataset_path_bias) == False:
        os.mkdir(torch_dataset_path_bias)
    
    torch_dataset_path_bias = torch_dataset_path_bias +'/' + model +'/'
    if os.path.exists(torch_dataset_path_bias) == False:
        os.mkdir(torch_dataset_path_bias)
    
    torch_dataset_path_bias = torch_dataset_path_bias + '/test/'
    if os.path.exists(torch_dataset_path_bias) == False:
        os.mkdir(torch_dataset_path_bias)
        
    torch_dataset_path_bias = torch_dataset_path_bias + '/bias_0/'
    if os.path.exists(torch_dataset_path_bias) == False:
        os.mkdir(torch_dataset_path_bias)
    
    file_names_bias = dataset_info.base_path + '/val/' + model + '/bias_0/'
    tot = 0
    for g in range(2,5+1):
        observation = g
        file_names = file_names_bias + 'observation_' + str(g) + '/'
        
        torch_dataset_path = torch_dataset_path_bias
            
        for l in range(dataset_info.scene_size_test[0],dataset_info.scene_size_test[1]+1):

            pt_dir = os.path.join(torch_dataset_path, f'{tot}.pt')

            
            image_dir = file_names + '/Scene' + str(l) + '/' + 'query.png'
            low_images = read_images(image_dir)
            
            image_dir = file_names + '/Scene' + str(l) + '/' + 'superimage.png'
            super_images = read_images(image_dir)
            
            #print(f' [-] converting scene {l} into {pt_dir}')
            convert_raw_to_numpy(dataset_info, low_images, super_images, pt_dir)
            print(f' [-] converting scene {l} into {pt_dir}')
            
            tot = tot+1
        
        
   


