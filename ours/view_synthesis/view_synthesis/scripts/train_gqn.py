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
from gqn import GenerativeQueryNetwork
from shepardmetzler import ShepardMetzler, transform_viewpoint
import collections
import numpy as np
from srgan import Discriminator

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

parser = argparse.ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
parser.add_argument('--gradient_steps', type=int, default=40000, help='number of gradient steps to run (default: 40000)')
parser.add_argument('--batch_size', type=int, default=8, help='size of batch (default: 112)')
parser.add_argument('--test_batch_size', type=int, default=1, help='size of batch (default: 1)')
parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=True)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--epochs', type=int, default=40000, metavar='N')
parser.add_argument('--log-interval', type=int, default=1, metavar='N')
parser.add_argument('--log_interval_record', type=int, default=1, metavar='N')

mse_function = nn.MSELoss(size_average=False)

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
        
def SaveRecord(data_dir,epoch,model,reconstruction_loss_train,kl_divergence_train,temp_loss_train,full_loss_train,
                       lRecord,hyper):
    directoy = data_dir
    fileName = directoy + '/model/model_' + str(epoch) + '.pkl'
    torch.save(model.state_dict(), fileName)
            
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'reconstruction_loss_train.txt'
    f = open(fileName, 'w')
    for l in range(len(reconstruction_loss_train)):
        f.write(str(reconstruction_loss_train[l]))
        f.write('\n')
    f.close()
            
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'kl_divergence_train.txt'
    f = open(fileName, 'w')
    for l in range(len(kl_divergence_train)):
        f.write(str(kl_divergence_train[l]))
        f.write('\n')
    f.close()
    
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'temp_loss_train.txt'
    f = open(fileName, 'w')
    for l in range(len(temp_loss_train)):
        f.write(str(temp_loss_train[l]))
        f.write('\n')
    f.close()
            
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'full_loss_train.txt'
    f = open(fileName, 'w')
    for l in range(len(full_loss_train)):
        f.write(str(full_loss_train[l]))
        f.write('\n')
    f.close()                        
    
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'lRecord.txt'
    f = open(fileName, 'w')
    for i in range(len(lRecord)):
        f.write(lRecord[i])
        f.write('\n')
    f.close()

    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'hyper.txt'
    f = open(fileName, 'w')
    for i in range(len(hyper)):
        f.write(str(hyper[i]))
        f.write('\n')
    f.close()


def train(model,train_loader,optimizer,epoch,lRecord):
    reconstruction_loss = AverageMeter()
    kl_divergence_loss = AverageMeter()
    temp_loss = AverageMeter()
    full_loss = AverageMeter()
    model.train()
    #for x, v in tqdm(loader):
    for batch_idx, (x1, x2, v1, v2) in enumerate(train_loader):
        if args.cuda:
            x1, x2, v1, v2 = x1.cuda(), x2.cuda(), v1.cuda(), v2.cuda()
        x1, x2, v1, v2 = Variable(x1), Variable(x2), Variable(v1), Variable(v2)
        
        #first frame################################################################
        frame = 0
        flag = True
        x_mu1, x_q1, x_o, r, kld, query_idx = model(x1, v1, flag,frame)
        
        
        ############################
        # Update GNQ network
        ###########################
        
        # If more than one GPU we must take new shape into account
        batch_size = x_q1.size(0)
        
        # Negative log likelihood
        nll = -Normal(x_mu1, sigma).log_prob(x_q1)
        
        reconstruction1 = torch.mean(nll.view(batch_size, -1), dim=0).sum()
        kl_divergence1  = torch.mean(kld.view(batch_size, -1), dim=0).sum()


        # Evidence lower bound
        elbo1 = reconstruction1 + kl_divergence1
        
        #second frame################################################################
        frame = query_idx
        flag = False
        x_mu2, x_q2, x_o, r, kld, query_idx = model(x2, v2, True,frame)
        
        # If more than one GPU we must take new shape into account
        batch_size = x_q2.size(0)
        
        # Negative log likelihood
        nll = -Normal(x_mu2, sigma).log_prob(x_q2)
        
        reconstruction2 = torch.mean(nll.view(batch_size, -1), dim=0).sum()
        kl_divergence2  = torch.mean(kld.view(batch_size, -1), dim=0).sum()


        # Evidence lower bound
        elbo2 = reconstruction2 + kl_divergence2
        
        tloss = ((x_mu1 - x_mu2)*(x_mu1 - x_mu2)).sum()/batch_size
        
        elbo = elbo1 + elbo2 + tloss
        
        elbo.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        reconstruction = reconstruction1 + reconstruction2
        kl_divergence = kl_divergence1 + kl_divergence2
        
        reconstruction_loss.update(reconstruction.item(), 1)
        kl_divergence_loss.update(kl_divergence.item(), 1)
        temp_loss.update(tloss.item(), 1)
        full_loss.update(elbo.item(), 1)
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Learning Rate: {}\t'
                  'Sigma: {}\t'
                  'GQN--\t'
                  'Reconctruction: {:.4f} ({:.4f}) \t'
                  'KL: {:.4f} ({:.4f}) \t'
                  'TP: {:.4f} ({:.4f}) \t'
                  'Full Loss: {:.4f} ({:.4f}) \t'.format(
                epoch, batch_idx * len(x1), len(train_loader.dataset),
                mu,
                sigma,
                reconstruction_loss.val, reconstruction_loss.avg,
                kl_divergence_loss.val, kl_divergence_loss.avg,
                temp_loss.val, temp_loss.avg,
                full_loss.val, full_loss.avg))
        lRecord.append('Train Epoch: {} [{}/{}]\t'
                  'Learning Rate: {}\t'
                  'Sigma: {}\t'
                  'GQN--\t'
                  'Reconctruction: {:.4f} ({:.4f}) \t'
                  'KL: {:.4f} ({:.4f}) \t'
                  'TP: {:.4f} ({:.4f}) \t'
                  'Full Loss: {:.4f} ({:.4f}) \t'.format(
                epoch, batch_idx * len(x1), len(train_loader.dataset),
                mu,
                sigma,
                reconstruction_loss.val, reconstruction_loss.avg,
                kl_divergence_loss.val, kl_divergence_loss.avg,
                temp_loss.val, temp_loss.avg,
                full_loss.val, full_loss.avg))
    return reconstruction_loss.avg, kl_divergence_loss.avg, temp_loss.avg, full_loss.avg

def main(step,dataset,data_dir):
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)    

    # Pixel variance
    global sigma_f, sigma_i
    sigma_f, sigma_i = 0.7, 2.0

    # Learning rate
    global mu_f, mu_i
    mu_f, mu_i = 5*10**(-5), 5*10**(-4)
    global mu, sigma
    mu, sigma = mu_i, sigma_i
    s = 0
    
    if step > 0:
        file_dir = data_dir + '/rate/model_' + str(step) + '_hyper.txt'
        temp = np.loadtxt(file_dir)
        mu = temp[0]
        sigma = temp[1]
        s = int(temp[2])

    # Create model and optimizer
    model = GenerativeQueryNetwork(x_dim=3, v_dim=5, r_dim=256, h_dim=128, z_dim=64, L=12)
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params GQN: {}'.format(n_parameters))
    
    # Model optimisations
    model.set_multiple_gpus()
    if step > 0:
        model_dir = data_dir+'/model/model_'+str(step)+'.pkl'
        model.load_state_dict(torch.load(model_dir))
    if args.cuda:
        model.cuda()

    cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=mu)

    # Load the dataset
    kwargs = {'num_workers':0, 'pin_memory': True} if args.cuda else {}
    
    train_loader_spatial = torch.utils.data.DataLoader(ShepardMetzler(root_dir=data_dir + '/torch' + '/train/'),#, target_transform=transform_viewpoint
                                         batch_size=args.batch_size,
                                         shuffle=True, **kwargs)

    lRecord = []
    reconstruction_loss_train = []
    kl_divergence_train = []
    temp_loss_train = []
    full_loss_train = []
    
    for epoch in range(step+1, args.epochs+step+1):
        hyper = []
        print('------------------Spatial-------------------------')
        train_loader = train_loader_spatial
        lRecord.append('------------------Spatial-------------------------')
        
        reconstruction_loss, kl_divergence,temp_loss, full_loss = train(model,train_loader,optimizer,epoch,lRecord)
        
        s = s+1
        reconstruction_loss_train.append(reconstruction_loss)
        kl_divergence_train.append(kl_divergence)
        temp_loss_train.append(temp_loss)
        full_loss_train.append(full_loss)
        
        # Anneal learning rate
        mu = max(mu_f + (mu_i - mu_f)*(1 - s/(1.6 * 10**6)), mu_f)
        for group in optimizer.param_groups:
            group["lr"] = mu * math.sqrt(1 - 0.999**s)/(1 - 0.9**s)
            
        # Anneal pixel variance
        sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - s/(2 * 10**5)), sigma_f)
        
        hyper.append(mu)
        hyper.append(sigma)
        hyper.append(s)
        
            
        if epoch % args.log_interval_record == 0:
            SaveRecord(data_dir,epoch,model,reconstruction_loss_train,kl_divergence_train,temp_loss_train,full_loss_train,
                       lRecord,hyper)

if __name__ == '__main__':
    dataset  = 'PVHM'
    data_dir = 'G:/2-paper/ResearchWork5/ResearchWork5_zz_finalversion/code/ours/view_synthesis/view_synthesis/data/' + dataset
    
    directoy = data_dir + '/model/'
    if os.path.exists(directoy) == False:
        os.mkdir(directoy)
        
    directoy = data_dir + '/rate/'
    if os.path.exists(directoy) == False:
        os.mkdir(directoy)
        
    directoy = data_dir + '/test/'
    if os.path.exists(directoy) == False:
        os.mkdir(directoy)
        
    step = 0
    main(step,dataset,data_dir)
