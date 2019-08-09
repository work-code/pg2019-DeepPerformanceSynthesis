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
Scene = collections.namedtuple('Scene', ['frames', 'cameras', 'superimage'])

dataset = 'PVHM'
set_type = 'val'
bias = 0
observation = 5
resolution = 128

_DATASETS = dict(#+ 'bias_' + str(bias) +'/observation_' + str(observation) +'/'
    DATA_SETTINGS=DatasetInfo(
        scene_path ='G:/2-paper/ResearchWork5/ResearchWork5_zz_finalversion/code/ours/view_synthesis/image_generation/data/'+dataset+'/gqndata/'+ str(resolution) + '/' + set_type +'/bias_' + str(bias) + '/observation_' + str(observation) + '/',
        base_path='G:/2-paper/ResearchWork5/ResearchWork5_zz_finalversion/code/ours/view_synthesis/image_generation/data/'+dataset,
        scene_size=[1,12000],
        frame_size=resolution,
        sequence_size=observation+1),
)
    
#views = [[-1., -1., 0., 0.707, 0.],
#        [-0.5, -1., 0., 1., 0.],
#        [0.5, -1., 0., 1., 0.],
#        [1., -1., 0., 0.707, 0.],
#        [1., -0.5, 0., -1., 0.],
#        [1., 0.5, 0., -1., 0.],
#        [1., 1., 0., -0.707, 0.],
#        [0.5, 1., 0., 0., 0.],
#        [-0.5, 1., 0., 0., 0.],
#        [-1., 1., 0., 0.707, 0.],
#        [-1., 0.5, 0., 1., 0.],
#        [-1, -0.5, 0., 1., 0.],
#        [-1., 1., 0., 0.707, 0.],
#        [-1., 0.5, 0., 1., 0.],
#        [-1, -0.5, 0., 1., 0.]]


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


def encapsulate(frames, cameras,superimage):
    return Scene(frames=frames,cameras=cameras, superimage = superimage)

def convert_raw_to_numpy(dataset_info, raw_images, view_points, super_images,pt_dir):
    scene = encapsulate(raw_images, view_points,super_images)
    with open(pt_dir, 'wb') as f:
        torch.save(scene, f)


def show_frame(frames, scene, views):
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.imshow(frames[scene,views])
    plt.show()

def read_images(scene_dir,sequence_size):
    raw_images = []
    for l in range(1,sequence_size+1):
        image_dir = scene_dir+ '/' + str(l)+'.jpg'
        with open(image_dir,'rb') as f:
            jpeg_data = f.read()
        #decoded_frames = tf.image.decode_jpeg(jpeg_data)
        #raw_image = tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)
        raw_images.append(jpeg_data)
    raw_images = numpy.array(raw_images)
    raw_images = raw_images.astype('object')    
    return raw_images

def read_images_super(scene_dir,sequence_size):
    raw_images = []
    image_dir = scene_dir+ '/super.jpg'
    with open(image_dir,'rb') as f:
        jpeg_data = f.read()
    raw_images.append(jpeg_data)
    raw_images = numpy.array(raw_images)
    raw_images = raw_images.astype('object')    
    return raw_images

def read_views(view_dir):
    raw_views = []
    scene_views = scio.loadmat(view_dir)
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
    torch_dataset = torch_dataset + '/' + 'bias_' + str(bias) + '/'
    if os.path.exists(torch_dataset) == False:
        os.mkdir(torch_dataset)
    torch_dataset = torch_dataset + '/' + 'observation_' + str(observation) + '/'
    if os.path.exists(torch_dataset) == False:
        os.mkdir(torch_dataset)
    
    file_names = dataset_info.scene_path
    tot = 0
    for l in range(dataset_info.scene_size[0],dataset_info.scene_size[1]+1):
        pt_dir = os.path.join(torch_dataset, f'{tot}.pt')
        tot = tot + 1
        
        scene_dir = file_names + '/scene/Scene' + str(l)
        raw_images = read_images(scene_dir,dataset_info.sequence_size)
        
        super_dir = file_names + '/scene/Scene' + str(l)
        super_images = read_images_super(scene_dir,dataset_info.sequence_size)
        
        view_dir = file_names + '/sceneview/Scene' + str(l) + '.mat'
        view_points = read_views(view_dir)
        
        print(f' [-] converting scene {l} into {scene_dir}')
        convert_raw_to_numpy(dataset_info, raw_images, view_points, super_images,pt_dir)
        
        


