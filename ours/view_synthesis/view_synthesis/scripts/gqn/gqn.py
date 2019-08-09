import random
import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from .representation import TowerRepresentation, PyramidRepresentation
from .generator import GeneratorNetwork


class GenerativeQueryNetwork(nn.Module):
    """
    Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: Number of refinements of density
    """
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L=12):
        super(GenerativeQueryNetwork, self).__init__()
        self.r_dim = r_dim

        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = TowerRepresentation(x_dim, v_dim, r_dim, pool=False)
        #self.representation = PyramidRepresentation(x_dim, v_dim, r_dim)

    def forward(self, images, viewpoints, flag, frame):
        """
        Forward through the GQN.

        :param images: batch of images [b, m, c, h, w]
        :param viewpoints: batch of viewpoints for image [b, m, k]
        """
        # Number of context datapoints to use for representation
        batch_size, m, *_ = viewpoints.size()

        # Sample random number of views and generate representation
        n_views = random.randint(2,5)
        indices = torch.randperm(m)
        indices = indices.sort()[0]
        
        if flag == True:
            query_idx = random.randint(0,m-1)
        else:
            query_idx = frame
        
        view_index = self.compute_view_index(n_views,m,query_idx)
        representation_idx = indices[view_index]
        
        #if flag == True:
        #    n_views = n_views + 1
        
        x, v = images[:, representation_idx], viewpoints[:, representation_idx]

        # Merge batch and view dimensions.
        _, _, *x_dims = x.size()
        _, _, *v_dims = v.size()

        x = x.view((-1, *x_dims))
        v = v.view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        phi = self.representation(x, v)

        # Seperate batch and view dimensions
        _, *phi_dims = phi.size()
        phi = phi.view((batch_size, n_views, *phi_dims))

        # sum over view representations
        r = torch.sum(phi, dim=1)

        # Use random (image, viewpoint) pair in batch as query
        x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]
        x_o = images[:, representation_idx]
        x_mu, kl = self.generator(x_q, v_q, r,True)

        # Return reconstruction and query viewpoint
        # for computing error
        return [x_mu, x_q, x_o, r, kl, query_idx]
    
    def compute_view_index(self,n_views,m,query_idx):
        view_bias = random.randint(0,m-1)
        step = math.ceil(m/n_views)
        start = 0
        view_index = [1] * (n_views)
        for g in range(0,n_views):
            index = start + step*g
            view_index[g] = index
        for g in range(0,n_views):
            view_index[g] = (view_index[g] + view_bias)%m
        view_index = sorted(view_index)
        #if flag == True:
        #    temp = [1] * (n_views+1)
        #    for g in range(0,n_views):
        #        temp[g] = view_index[g]
        #    temp[n_views] = query_idx
        #    return temp
        #if flag == False:
        return view_index

    def sample(self, context_x, context_v, viewpoint, sigma):
        """
        Sample from the network given some context and viewpoint.

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        :param viewpoint: viewpoint to generate image from
        :param sigma: pixel variance
        """
        batch_size, n_views, _, h, w = context_x.size()
        
        _, _, *x_dims = context_x.size()
        _, _, *v_dims = context_v.size()

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        phi = self.representation(x, v)

        _, *phi_dims = phi.size()
        phi = phi.view((batch_size, n_views, *phi_dims))

        r = torch.sum(phi, dim=1)

        x_mu = self.generator.sample((h, w), viewpoint, r)
        x_sample = Normal(x_mu, sigma).sample()
        return x_sample
        
    def set_multiple_gpus(self):
        # here uses multi gpu
        self.generator = nn.DataParallel(self.generator).cuda()
        self.representation = nn.DataParallel(self.representation).cuda()