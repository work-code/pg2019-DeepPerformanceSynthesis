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
    ['base_path', 'train_scene_size', 'test_scene_size','frame_size', 'sequence_size']
)
Scene = collections.namedtuple('Scene', ['low_images', 'super_images'])

_DATASETS = dict(
    fruits=DatasetInfo(
        base_path='G:/2-paper/ResearchWork5/ResearchWork5_zz_finalversion/code/ours/super_resolution/DataConvert/data/' + 'PVHM',
        train_scene_size=[1,400],
        test_scene_size=[1,200],
        frame_size=128,
        sequence_size=24),
)


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

    torch_dataset_path = dataset_info.base_path + '/torch'
    torch_dataset_path_train = f'{torch_dataset_path}/train'
    torch_dataset_path_test = f'{torch_dataset_path}/test'

    if os.path.exists(torch_dataset_path) == False:
        os.mkdir(torch_dataset_path)
    if os.path.exists(torch_dataset_path_train) == False:
        os.mkdir(torch_dataset_path_train)
    if os.path.exists(torch_dataset_path_test) == False:
        os.mkdir(torch_dataset_path_test)
    
    file_names = dataset_info.base_path + '/superdata/train'
    tot = 0
    for l in range(dataset_info.train_scene_size[0],dataset_info.train_scene_size[1]+1):
        
        for t in range(1,dataset_info.sequence_size+1):
            pt_dir = os.path.join(torch_dataset_path_train, f'{tot}.pt')
            tot = tot + 1
            
            image_dir = file_names + '/scene/Scene' + str(l) + '/' + str(t) + '.jpg'
            low_images = read_images(image_dir)
            
            image_dir = file_names + '/superscene/Scene' + str(l) + '/' + str(t) + '.jpg'
            super_images = read_images(image_dir)
            
            #print(f' [-] converting scene {l} into {pt_dir}')
            convert_raw_to_numpy(dataset_info, low_images, super_images, pt_dir)
        print(f' [-] converting scene {l} into {pt_dir}')
        
        
    print(f' [-] {tot} scenes in the train dataset')
    
    file_names = dataset_info.base_path + '/superdata/test'
    tot = 0
    for l in range(dataset_info.test_scene_size[0],dataset_info.test_scene_size[1]+1):
        
        for t in range(1,dataset_info.sequence_size+1):
            pt_dir = os.path.join(torch_dataset_path_test, f'{tot}.pt')
            tot = tot + 1
            
            image_dir = file_names + '/scene/Scene' + str(l) + '/' + str(t) + '.jpg'
            low_images = read_images(image_dir)
            
            image_dir = file_names + '/superscene/Scene' + str(l) + '/' + str(t) + '.jpg'
            super_images = read_images(image_dir)
            
            #print(f' [-] converting scene {l} into {pt_dir}')
            convert_raw_to_numpy(dataset_info, low_images, super_images, pt_dir)
        print(f' [-] converting scene {l} into {pt_dir}')



