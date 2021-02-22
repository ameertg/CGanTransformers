#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from cgan_utils import ReplayBuffer
from cgan_utils import LambdaLR
from data_utils import *

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator()
netG_B2A = Generator()
netD_A = Discriminator()
netD_B = Discriminator()
netC = CycleLoss()

cycle_loss = CycleLoss()

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

# Lossess
criterion_GAN = torch.nn.CrossEntropyLoss()
criterion_cycle = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_cycle = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_recon = torch.optim.lr_scheduler.LambdaLR(optimizer_cycle, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

nes_corpus = get_lm_corpus('data/nesmdb_tx1', 'nesmdb')
nes_corpus_iter = nes_corpus.get_iterator('train', bsz=1, bptt=12)
lakh_corpus_iter = nes_corpus_iter

# Loss plot
###################################

###### Training ######
for epoch in range(0, 100):
    for i, (data, _, bptt) in enumerate(tqdm(nes_corpus_iter)):
        # Set model input
        real_A = data.clone().detach()
        # Change this to include other dataset
        real_B =  data.clone().detach()


        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = netG_A2B(real_A, bptt)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake.float(), target_real.float())

        fake_A = netG_B2A(real_B, bptt)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake.float(), target_real.float())

        # Cycle loss
        recovered_A = netG_B2A(fake_B, bptt)
        loss_cycle_ABA = netC(real_A, recovered_A)*10.0

        recovered_B = netG_A2B(fake_A, bptt)
        loss_cycle_BAB = netC(real_A, recovered_A)*10.0

        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        ###### Cycle loss ######
        optimizer_cycle.zero_grad()

        # A-A
        pred_real = CycleLoss(real_A.detach(), real_A.detach())
        loss_real_A = criterion_cycle(pred_real, target_real)

        # A-A*
        pred_fake = CycleLoss(real_A.detach(), fake_A.detach())
        loss_fake_A = criterion_cycle(pred_fake, target_fake)

        # B-B
        pred_real = CycleLoss(real_B.detach(), real_B.detach())
        loss_real_B = criterion_cycle(pred_real, target_real)

        # B-B*
        pred_fake = CycleLoss(real_B.detach(), fake_B.detach())
        loss_fake_B = criterion_cycle(pred_fake, target_fake)

        # Total loss
        loss_cycle = (loss_real_A + loss_real_B + loss_fake_A + loss_fake_B)*0.25
        loss_cycle.backward()

        optimizer_cycle.step()
        ###################################


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')
    torch.save(netC.state_dict(), 'output/netC.pth')
###################################
