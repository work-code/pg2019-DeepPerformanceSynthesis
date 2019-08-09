import collections
import os, io
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

Scene = collections.namedtuple('Scene', ['low_images', 'super_images'])



class ShepardMetzler(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
        data = torch.load(scene_path)

        byte_to_tensor = lambda x: ToTensor()(Image.open(io.BytesIO(x)))

        low_images = torch.stack([byte_to_tensor(frame) for frame in data.low_images])
        super_images = torch.stack([byte_to_tensor(frame) for frame in data.super_images])

        low_images = low_images[0,:,:,:]
        super_images = super_images[0,:,:,:]
        
        if self.transform:
            low_images = self.transform(low_images)
            
        if self.target_transform:
            super_images = self.transform(super_images)

        return low_images, super_images