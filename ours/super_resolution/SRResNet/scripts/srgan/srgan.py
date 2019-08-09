import math

import torch.nn.functional as F
from torch import nn

from .discriminator import Discriminator
from .generator import Generator


class GeneratorMM(nn.Module):

    def __init__(self, scale_factor):
        super(GeneratorMM, self).__init__()
        self.generator = Generator(scale_factor)
    def forward(self, x):
        x = self.generator(x)
        return x
    
    def set_multiple_gpus(self):
        # here uses multi gpu
        self.generator = nn.DataParallel(self.generator).cuda()
        
class DiscriminatorMM(nn.Module):

    def __init__(self):
        super(DiscriminatorMM, self).__init__()
        self.discriminator = Discriminator()
    
    def forward(self, x):
        x = self.discriminator(x)
        return x
    
    def set_multiple_gpus(self):
        # here uses multi gpu
        self.discriminator = nn.DataParallel(self.discriminator).cuda()



