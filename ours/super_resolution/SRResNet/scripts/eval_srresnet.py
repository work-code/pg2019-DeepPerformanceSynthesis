import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from loss import GeneratorLoss
from srgan import GeneratorMM,DiscriminatorMM
from torchvision import models
import torch.utils.model_zoo as model_zoo
from shepardmetzler import ShepardMetzler
import collections
from torchvision.utils import save_image
import numpy, scipy.io

Scene = collections.namedtuple('Scene', ['low_images', 'super_images'])

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument('--eval_batch_size', type=int, default=1, help='size of batch (default: 1)')
parser.add_argument('--epochs', type=int, default=1000000, metavar='N')
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument('--log_interval', type=int, default=20, metavar='N')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')

    
def evaluation(netG,netD,eval_loader,eval_dir):
    
    netG.eval()
    netD.eval()
    with torch.no_grad():
        for batch_idx, (low_images, real_img) in enumerate(eval_loader):
            if args.cuda:
                low_images, real_img = low_images.cuda(), real_img.cuda()
            low_images, real_img = Variable(low_images), Variable(real_img)
            
            fake_img = netG(low_images)
            
            scene_name = 'scene' + str(batch_idx + 1)
            scene_dir = eval_dir + 'scene' + str(batch_idx + 1) + '/'
            if os.path.exists(scene_dir) == False:
                os.mkdir(scene_dir)
            
            save_image(fake_img[0,:,:,:].float(), scene_dir + 'generation.png')
            save_image(low_images[0,:,:,:].float(), scene_dir + 'low_image.png')
            save_image(real_img[0,:,:,:].float(), scene_dir + 'super_image.png')
        
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            
            disc_image = [real_out.item(),fake_out.item()]
            scipy.io.savemat(scene_dir +'disc_image.mat', mdict={'disc_image': disc_image})
            print(scene_name)
            
            
def main(step,data_bias,data_dir,directoy):

    global args, model, netContent

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed) 
        
    netG = GeneratorMM(args.upscale_factor)
    netD = DiscriminatorMM()
    netG.set_multiple_gpus()
    netD.set_multiple_gpus()
    if step > 0:
        model_dir = data_bias+'/model/modelG_'+str(step)+'.pkl'
        netG.load_state_dict(torch.load(model_dir))
        
        model_dir = data_bias+'/model/modelD_'+str(step)+'.pkl'
        netD.load_state_dict(torch.load(model_dir))
    if args.cuda:
        netG = netG.cuda()
        netD = netD.cuda()

    cudnn.benchmark = True

    # Load the dataset
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    eval_loader = torch.utils.data.DataLoader(ShepardMetzler(root_dir=data_dir),
                                         batch_size = args.eval_batch_size,
                                         shuffle=False, **kwargs)

   
    eval_dir = directoy
            
    evaluation(netG,netD,eval_loader,eval_dir)          


if __name__ == "__main__":
    step = 1
    model = 'model_28400'
    dataset  = 'PVHM'
    data_bias = 'G:/2-paper/ResearchWork5/ResearchWork5_zz_finalversion/code/ours/super_resolution/SRResNet/data/' + dataset +'/'
    
    directoy_bias = data_bias + '/eval_raw/'
    if os.path.exists(directoy_bias) == False:
        os.mkdir(directoy_bias)
    
    directoy_bias = directoy_bias + '/model_' + str(step) + '/'
    if os.path.exists(directoy_bias) == False:
        os.mkdir(directoy_bias)
    
    directoy_bias = directoy_bias + '/bias_0/'
    if os.path.exists(directoy_bias) == False:
        os.mkdir(directoy_bias)
    
    data_dir_bias = data_bias + '/' + 'torch_super/' + model + '/val/bias_0/'
    for g in range(2,5+1):
        data_dir = data_dir_bias + '/observation_' + str(g) + '/'
        directoy = directoy_bias + '/observation_' + str(g) + '/'
        if os.path.exists(directoy) == False:
            os.mkdir(directoy)
        main(step,data_bias,data_dir,directoy)
