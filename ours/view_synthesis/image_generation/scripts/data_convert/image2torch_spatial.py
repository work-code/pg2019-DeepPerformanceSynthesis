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
#import tensorflow as tf
import scipy.io as scio

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['base_path', 'scene_path','scene_size','frame_size', 'sequence_size']
)
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

dataset = 'PVHM'
set_type = 'train'
resolution = 128

_DATASETS = dict(
    DATA_SETTINGS=DatasetInfo(
        base_path='G:/2-paper/ResearchWork5/ResearchWork5_zz_finalversion/code/ours/view_synthesis/image_generation/data/'+dataset+'/',
        scene_path= 'G:/2-paper/ResearchWork5/ResearchWork5_zz_finalversion/code/ours/view_synthesis/image_generation/data/'+dataset+'/'+'gqndata/' + str(resolution) + '/'+set_type+'/',
        scene_size=[1,400],
        frame_size=resolution,
        sequence_size=20),
)
    


def encapsulate(frames, cameras):
    return Scene(cameras=cameras, frames=frames)

def convert_raw_to_numpy(dataset_info, raw_images, view_points, pt_dir):
    scene = encapsulate(raw_images, view_points)
    with open(pt_dir, 'wb') as f:
        torch.save(scene, f)


def show_frame(frames, scene, views):
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.imshow(frames[scene,views])
    plt.show()

def read_images(scene_dir,scene_dir_2,sequence_size):
    raw_images = []
    for l in range(1,sequence_size+1):
        image_dir = scene_dir+ '/' + str(l)+'.jpg'
        with open(image_dir,'rb') as f:
            jpeg_data = f.read()
        #decoded_frames = tf.image.decode_jpeg(jpeg_data)
        #raw_image = tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)
        raw_images.append(jpeg_data)
    
    for l in range(1,sequence_size+1):
        image_dir = scene_dir_2+ '/' + str(l)+'.jpg'
        with open(image_dir,'rb') as f:
            jpeg_data = f.read()
        #decoded_frames = tf.image.decode_jpeg(jpeg_data)
        #raw_image = tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)
        raw_images.append(jpeg_data)
    raw_images = numpy.array(raw_images)
    raw_images = raw_images.astype('object')    
    return raw_images

def read_views(view_dir,view_dir_2):
    raw_views = []
    scene_views = scio.loadmat(view_dir)
    scene_views = scene_views["shape"]
    scene_views = scene_views[:,0:5]
    
    raw_views.append(scene_views)
    
    scene_views = scio.loadmat(view_dir_2)
    scene_views = scene_views["shape"]
    scene_views = scene_views[:,0:5]
    
    raw_views.append(scene_views)
    raw_views = numpy.array(raw_views)
    raw_views = raw_views.astype('float32')
    return raw_views

if __name__ == '__main__':

    DATASET = 'DATA_SETTINGS'
    dataset_info = _DATASETS[DATASET]

    torch_dataset = dataset_info.base_path + '/torch/'
    if os.path.exists(torch_dataset) == False:
        os.mkdir(torch_dataset)
    torch_dataset = torch_dataset + '/' + set_type + '/'
    if os.path.exists(torch_dataset) == False:
        os.mkdir(torch_dataset)
    
    file_names = dataset_info.scene_path
    tot = 0
    for l in range(dataset_info.scene_size[0],dataset_info.scene_size[1]):
        pt_dir = os.path.join(torch_dataset, f'{tot}.pt')
        tot = tot + 1
        
        scene_dir = file_names + '/scene/Scene' + str(l)
        scene_dir_2 = file_names + '/scene/Scene' + str(l+1)
        raw_images = read_images(scene_dir,scene_dir_2,dataset_info.sequence_size)
        
        view_dir = file_names + '/sceneview/Scene' + str(l) + '.mat'
        view_dir_2 = file_names + '/sceneview/Scene' + str(l) + '.mat'
        view_points = read_views(view_dir,view_dir_2)
        
        print(f' [-] converting scene {l} into {scene_dir}')
        convert_raw_to_numpy(dataset_info, raw_images, view_points, pt_dir)
        
        


