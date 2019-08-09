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


Scene = collections.namedtuple('Scene', ['low_images', 'super_images'])

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument('--batch_size', type=int, default=1, help='size of batch (default: 36)')#36 18
parser.add_argument('--test_batch_size', type=int, default=1, help='size of batch (default: 1)')
parser.add_argument('--epochs', type=int, default=20000, metavar='N')
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--vgg_loss", action="store_true", default=True, help="Use content loss?")
parser.add_argument('--log_interval', type=int, default=1, metavar='N')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = args.lr
    return lr 

def SaveRecord(data_dir,epoch,netG, netD,generator_loss_train,a_loss_train,p_loss_train,i_loss_train,t_loss_train,discriminator_loss_train,
                       generator_loss_test,a_loss_test,p_loss_test,i_loss_test,t_loss_test,discriminator_loss_test,lRecord):
    directoy = data_dir
    fileName = directoy + '/model/modelG_' + str(epoch) + '.pkl'
    torch.save(netG.state_dict(), fileName)
    
    fileName = directoy + '/model/modelD_' + str(epoch) + '.pkl'
    torch.save(netD.state_dict(), fileName)
            
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'generator_loss_train.txt'
    f = open(fileName, 'w')
    for l in range(len(generator_loss_train)):
        f.write(str(generator_loss_train[l]))
        f.write('\n')
    f.close()
    #adversarial_loss,perception_loss,image_loss,tv_loss
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'adversarial_loss_train.txt'
    f = open(fileName, 'w')
    for l in range(len(a_loss_train)):
        f.write(str(a_loss_train[l]))
        f.write('\n')
    f.close()
    
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'perception_loss_train.txt'
    f = open(fileName, 'w')
    for l in range(len(p_loss_train)):
        f.write(str(p_loss_train[l]))
        f.write('\n')
    f.close()
    
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'image_loss_train.txt'
    f = open(fileName, 'w')
    for l in range(len(i_loss_train)):
        f.write(str(i_loss_train[l]))
        f.write('\n')
    f.close()
    
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'tv_loss_train.txt'
    f = open(fileName, 'w')
    for l in range(len(t_loss_train)):
        f.write(str(t_loss_train[l]))
        f.write('\n')
    f.close()
            
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'discriminator_loss_train.txt'
    f = open(fileName, 'w')
    for l in range(len(discriminator_loss_train)):
        f.write(str(discriminator_loss_train[l]))
        f.write('\n')
    f.close()
            
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'generator_loss_test.txt'
    f = open(fileName, 'w')
    for l in range(len(generator_loss_test)):
        f.write(str(generator_loss_test[l]))
        f.write('\n')
    f.close()
    
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'adversarial_loss_test.txt'
    f = open(fileName, 'w')
    for l in range(len(a_loss_test)):
        f.write(str(a_loss_test[l]))
        f.write('\n')
    f.close()
    
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'perception_loss_test.txt'
    f = open(fileName, 'w')
    for l in range(len(p_loss_test)):
        f.write(str(p_loss_test[l]))
        f.write('\n')
    f.close()
    
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'image_loss_test.txt'
    f = open(fileName, 'w')
    for l in range(len(i_loss_test)):
        f.write(str(i_loss_test[l]))
        f.write('\n')
    f.close()
    
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'tv_loss_test.txt'
    f = open(fileName, 'w')
    for l in range(len(t_loss_test)):
        f.write(str(t_loss_test[l]))
        f.write('\n')
    f.close()
            
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'discriminator_loss_test.txt'
    f = open(fileName, 'w')
    for l in range(len(discriminator_loss_test)):
        f.write(str(discriminator_loss_test[l]))
        f.write('\n')
    f.close()
            
    fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'lRecord.txt'
    f = open(fileName, 'w')
    for i in range(len(lRecord)):
        f.write(lRecord[i])
        f.write('\n')
    f.close()
    
def test(netG, netD,start,test_loader,epoch,generator_criterion,lRecord,test_dir):
    discriminator_loss = AverageMeter()
    generator_loss = AverageMeter()
    a_loss = AverageMeter()
    p_loss = AverageMeter()
    i_loss = AverageMeter()
    t_loss = AverageMeter()
    
    netG.eval()
    netD.eval()
    with torch.no_grad():
        for batch_idx, (low_images, real_img) in enumerate(test_loader,start):
            if args.cuda:
                low_images, real_img = low_images.cuda(), real_img.cuda()
            low_images, real_img = Variable(low_images), Variable(real_img)
                    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            fake_img = netG(low_images)
            
            save_image(fake_img[0,:,:,:].float(), test_dir + 'generation.png')
            save_image(low_images[0,:,:,:].float(), test_dir + 'low_image.png')
            save_image(real_img[0,:,:,:].float(), test_dir + 'super_image.png')
        
            real_label = netD(real_img).mean()
            fake_label = netD(fake_img).mean()
            
            d_loss = 1 - real_label + fake_label
            #d_loss = torch.mean(-(torch.log(1-fake_label) + torch.log(real_label)))
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            g_loss,adversarial_loss,perception_loss,image_loss,tv_loss = generator_criterion(fake_label, fake_img, real_img)
        
            discriminator_loss.update(d_loss.item(), 1)
            generator_loss.update(g_loss.item(), 1)
        
            a_loss.update(adversarial_loss.item(), 1)
            p_loss.update(perception_loss.item(), 1)
            i_loss.update(image_loss.item(), 1)
            t_loss.update(tv_loss.item(), 1)
            
            if batch_idx % args.log_interval_test == 0:
                print('Test epoch: {} [{}/{}]\t'
                      'Learning rate: {}\t'
                      'Generator loss: {:.4f} ({:.4f}) \t'
                      'Adversarial loss: {:.4f} ({:.4f}) \t'
                      'Perception loss: {:.4f} ({:.4f}) \t'
                      'Image loss: {:.4f} ({:.4f}) \t'
                      'Tv loss: {:.4f} ({:.4f}) \t'
                      'Discriminator loss: {:.2f} ({:.2f}) \t'.format(
                              epoch, batch_idx * len(low_images), len(test_loader.dataset),
                              lr,
                              generator_loss.val, generator_loss.avg,
                              a_loss.val,a_loss.avg,
                              p_loss.val,p_loss.avg,
                              i_loss.val,i_loss.avg,
                              t_loss.val,t_loss.avg,
                              discriminator_loss.val, discriminator_loss.avg))
            lRecord.append('Test epoch: {} [{}/{}]\t'
                      'Learning rate: {}\t'
                      'Generator loss: {:.4f} ({:.4f}) \t'
                      'Adversarial loss: {:.4f} ({:.4f}) \t'
                      'Perception loss: {:.4f} ({:.4f}) \t'
                      'Image loss: {:.4f} ({:.4f}) \t'
                      'Tv loss: {:.4f} ({:.4f}) \t'
                      'Discriminator loss: {:.2f} ({:.2f}) \t'.format(
                              epoch, batch_idx * len(low_images), len(test_loader.dataset),
                              lr,
                              generator_loss.val, generator_loss.avg,
                              a_loss.val,a_loss.avg,
                              p_loss.val,p_loss.avg,
                              i_loss.val,i_loss.avg,
                              t_loss.val,t_loss.avg,
                              discriminator_loss.val, discriminator_loss.avg))
            break
    return generator_loss.avg,a_loss.avg,p_loss.avg,i_loss.avg,t_loss.avg, discriminator_loss.avg

def train(train_loader, optimizerG, optimizerD, netG, netD, generator_criterion, epoch,lRecord):
    
    discriminator_loss = AverageMeter()
    generator_loss = AverageMeter()
    a_loss = AverageMeter()
    p_loss = AverageMeter()
    i_loss = AverageMeter()
    t_loss = AverageMeter()
    
    netG.train()
    netD.train()
    for batch_idx, (low_images, real_img) in enumerate(train_loader):

        if args.cuda:
            low_images, real_img = low_images.cuda(), real_img.cuda()
        low_images, real_img = Variable(low_images), Variable(real_img)
        
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        fake_img = netG(low_images)
            
        netD.zero_grad()
        real_label = netD(real_img).mean()
        fake_label = netD(fake_img).mean()
        
        d_loss = 1 - real_label + fake_label
        #d_loss = torch.mean(-(torch.log(1-fake_label) + torch.log(real_label)))
        
        d_loss.backward(retain_graph=True)
        optimizerD.step()
        
        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss,adversarial_loss,perception_loss,image_loss,tv_loss = generator_criterion(fake_label, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        
        #fake_img = netG(low_images)
        #fake_out = netD(fake_img).mean()

        #g_loss = generator_criterion(fake_out, fake_img, real_img)
        #d_loss = 1 - real_out + fake_out
        
        discriminator_loss.update(d_loss.item(), 1)
        generator_loss.update(g_loss.item(), 1)
        
        a_loss.update(adversarial_loss.item(), 1)
        p_loss.update(perception_loss.item(), 1)
        i_loss.update(image_loss.item(), 1)
        t_loss.update(tv_loss.item(), 1)
        
        
        if batch_idx % args.log_interval == 0:
            print('Train epoch: {} [{}/{}]\t'
                  'Learning rate: {}\t'
                  'Generator loss: {:.4f} ({:.4f}) \t'
                  'Adversarial loss: {:.4f} ({:.4f}) \t'
                  'Perception loss: {:.4f} ({:.4f}) \t'
                  'Image loss: {:.4f} ({:.4f}) \t'
                  'Tv loss: {:.4f} ({:.4f}) \t'
                  'Discriminator loss: {:.2f} ({:.2f}) \t'.format(
                epoch, batch_idx * len(low_images), len(train_loader.dataset),
                lr,
                generator_loss.val, generator_loss.avg,
                a_loss.val,a_loss.avg,
                p_loss.val,p_loss.avg,
                i_loss.val,i_loss.avg,
                t_loss.val,t_loss.avg,
                discriminator_loss.val, discriminator_loss.avg))
        lRecord.append('Train epoch: {} [{}/{}]\t'
                  'Learning rate: {}\t'
                  'Generator loss: {:.4f} ({:.4f}) \t'
                  'Adversarial loss: {:.4f} ({:.4f}) \t'
                  'Perception loss: {:.4f} ({:.4f}) \t'
                  'Image loss: {:.4f} ({:.4f}) \t'
                  'Tv loss: {:.4f} ({:.4f}) \t'
                  'Discriminator loss: {:.2f} ({:.2f}) \t'.format(
                epoch, batch_idx * len(low_images), len(train_loader.dataset),
                lr,
                generator_loss.val, generator_loss.avg,
                a_loss.val,a_loss.avg,
                p_loss.val,p_loss.avg,
                i_loss.val,i_loss.avg,
                t_loss.val,t_loss.avg,
                discriminator_loss.val, discriminator_loss.avg))
        
    return generator_loss.avg,a_loss.avg,p_loss.avg,i_loss.avg,t_loss.avg, discriminator_loss.avg

def main(step,dataset,data_dir):

    global args, model, netContent,lr
    
    args = parser.parse_args()
    lr = args.lr
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed) 
        
    netG = GeneratorMM(args.upscale_factor)
    n_parameters = sum([p.data.nelement() for p in netG.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    
    netD = DiscriminatorMM()
    n_parameters = sum([p.data.nelement() for p in netD.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    
    generator_criterion = GeneratorLoss()
    netG.set_multiple_gpus()
    netD.set_multiple_gpus()
    if step > 0:
        model_dir = data_dir+'/model/modelG_'+str(step)+'.pkl'
        netG.load_state_dict(torch.load(model_dir))
        
        model_dir = data_dir+'/model/modelD_'+str(step)+'.pkl'
        netD.load_state_dict(torch.load(model_dir))
    if args.cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        generator_criterion = generator_criterion.cuda()
    cudnn.benchmark = True
    
    optimizerG = optim.Adam(netG.parameters(),lr = args.lr,betas=(0.9, 0.999))
    optimizerD = optim.Adam(netD.parameters(),lr = args.lr,betas=(0.9, 0.999))

    # Load the dataset
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    
    train_loader = torch.utils.data.DataLoader(ShepardMetzler(root_dir=data_dir + '/torch' + '/train/'),
                                         batch_size=args.batch_size,
                                         shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(ShepardMetzler(root_dir=data_dir + '/torch' + '/test/'),
                                         batch_size = args.test_batch_size,
                                         shuffle=False, **kwargs)


    lRecord = []
    generator_loss_train = []
    discriminator_loss_train = []
    a_loss_train = []
    p_loss_train = []
    i_loss_train = []
    t_loss_train = []
    
    generator_loss_test = []
    discriminator_loss_test = []
    a_loss_test = []
    p_loss_test = []
    i_loss_test = []
    t_loss_test = []
    start = 0
    for epoch in range(step+1, args.epochs+step+1):
        generator_loss,a_loss,p_loss,i_loss,t_loss, discriminator_loss = train(train_loader, optimizerG, optimizerD, netG, netD, generator_criterion, epoch,lRecord)
        generator_loss_train.append(generator_loss)
        a_loss_train.append(a_loss)
        p_loss_train.append(p_loss)
        i_loss_train.append(i_loss)
        t_loss_train.append(t_loss)
        discriminator_loss_train.append(discriminator_loss)
        
        
        lr = adjust_learning_rate(optimizerG, epoch-1)
        for param_group in optimizerG.param_groups:
            param_group["lr"] = lr
    
        lr = adjust_learning_rate(optimizerD, epoch-1)
        for param_group in optimizerD.param_groups:
            param_group["lr"] = lr
            
        if epoch % args.log_interval_test == 0:
            test_dir = data_dir + '/test/' + 'model' +str(epoch) + '_scene' + str(start + 1) + '/'
            if os.path.exists(test_dir) == False:
                os.mkdir(test_dir)

            generator_loss,a_loss,p_loss,i_loss,t_loss, discriminator_loss = test(netG, netD,start,test_loader,epoch,generator_criterion,lRecord,test_dir)
            start = (start + 1)%len(test_loader)
            generator_loss_test.append(generator_loss)
            a_loss_test.append(a_loss)
            p_loss_test.append(p_loss)
            i_loss_test.append(i_loss)
            t_loss_test.append(t_loss)
            discriminator_loss_test.append(discriminator_loss)
            
        if epoch % args.log_interval_record == 0:
            SaveRecord(data_dir,epoch,netG, netD,generator_loss_train,a_loss_train,p_loss_train,i_loss_train,t_loss_train,discriminator_loss_train,
                       generator_loss_test,a_loss_test,p_loss_test,i_loss_test,t_loss_test,discriminator_loss_test,lRecord)



if __name__ == "__main__":
    dataset  = 'PVHM'
    data_dir = 'G:/2-paper/ResearchWork5/ResearchWork5_zz_finalversion/code/ours/super_resolution/SRResNet/data/' + dataset
    
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
