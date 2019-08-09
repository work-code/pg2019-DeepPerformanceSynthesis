import os
import sys
import random
import math
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from gqnval import GenerativeQueryNetworkVal
from shepardmetzler import ShepardMetzler, transform_viewpoint
import collections

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])
scene_name = ['action41','action42','action43','action44','action45','action46','action47','action48','action49','action50']
view_num = 20

parser = argparse.ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
parser.add_argument('--gradient_steps', type=int, default=2*(10**6), help='number of gradient steps to run (default: 2 million)')
parser.add_argument('--val_batch_size', type=int, default=1, help='size of batch (default: 1)')
parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--epochs', type=int, default=1000000, metavar='N')
parser.add_argument('--log-interval', type=int, default=5, metavar='N')
parser.add_argument('--log_interval_test', type=int, default=1, metavar='N')
parser.add_argument('--log_interval_record', type=int, default=1, metavar='N')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def valTemporal(model,val_loader,val_dir,observations):

    val_dir = val_dir + str(observations) +'/'
    if os.path.exists(val_dir) == False:
        os.mkdir(val_dir)
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, v) in enumerate(val_loader):
            if args.cuda:
                x, v = x.cuda(), v.cuda()
            x, v = Variable(x), Variable(v)
            x_mu, x_q, x_o, r, kld,representation_idx = model(x, v, observations)

            save_image(x_mu[0,:,:,:].float(), val_dir + '/query' + str(batch_idx) + '.jpg')
            save_image(x_q[0,:,:,:].float(), val_dir + '/groundtruth' + str(batch_idx) + '.jpg')
            x_o = x_o[0,:,:,:,:]
            for l in range(0,len(x_o)):
                save_image(x_o[l,:,:,:].float(), val_dir + '/observation_'+str((representation_idx[l].numpy()))+'_'+str(batch_idx)+'.jpg')



def main(step,dataset,data_dir):
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)    

    # Create model and optimizer
    model = GenerativeQueryNetworkVal(x_dim=3, v_dim=6, r_dim=256, h_dim=128, z_dim=64, L=12)
    model.set_multiple_gpus()
    if step > 0:
        model_dir = data_dir+'/model/model_'+str(step)+'.pkl'
        model.load_state_dict(torch.load(model_dir))
    if args.cuda:
        model.cuda()
    cudnn.benchmark = True
    
    # Model optimisations
    #model = nn.DataParallel(model) if args.data_parallel else model
    #model = model.half() if args.fp16 else model

   

    # Load the dataset
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    tot = 0
    for t in range(0,len(scene_name)):
        if t == 1:
            return

        for g in range(0,view_num): 
            data_root = data_dir + '/torch' + '/val/temporal/' + scene_name[t] + '/view' + str(g) + '/'
            val_loader = torch.utils.data.DataLoader(ShepardMetzler(root_dir=data_root),#, target_transform=transform_viewpoin
                                                     batch_size = args.val_batch_size,
                                                     shuffle=False, **kwargs)
            val_dir = data_dir + '/val/' 
            if os.path.exists(val_dir) == False:
                os.mkdir(val_dir)
            val_dir = val_dir + '/temporal/' 
            if os.path.exists(val_dir) == False:
                os.mkdir(val_dir)
            val_dir = val_dir + '/' + scene_name[t] + '/' 
            if os.path.exists(val_dir) == False:
                os.mkdir(val_dir)

            val_dir = val_dir + '/' +'view' + str(g) + '/' 
            if os.path.exists(val_dir) == False:
                os.mkdir(val_dir)
            for m in range(1,2):
                print('------------'+ 'scene'+ str(t)+'---'+'view' + str(g)+'---'+'observation' + str(m)+'--------------')
                valTemporal(model,val_loader,val_dir,m)

if __name__ == '__main__':
    dataset  = 'PVHM'
    data_dir = 'E:/3_researchWork5/VideoMorphing/GQN/data/' + dataset
    
    directoy = data_dir + '/model/'
    if os.path.exists(directoy) == False:
        os.mkdir(directoy)
        
    directoy = data_dir + '/rate/'
    if os.path.exists(directoy) == False:
        os.mkdir(directoy)
        
    directoy = data_dir + '/test/'
    if os.path.exists(directoy) == False:
        os.mkdir(directoy)
        
    step = 240
    main(step,dataset,data_dir)
