#!/usr/bin/python3

import argparse
import itertools

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


from models import Generator, Discriminator, CycleLoss
from cgan_utils import LambdaLR
from data_utils import *
from utils.exp_utils import save_checkpointFull

from tqdm import trange


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=10, help='size of the batches')
parser.add_argument('--nes', type=str, default='data/nesmdb_tx1', help='root directory of the dataset')
parser.add_argument('--lakh', type=str, default='data/5k_poprock_tx1', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--gen_train', type=int, default=1, help='number of iterations to wait before training the generator')
parser.add_argument('--temp', type=int, default=0.1, help='number of iterations to wait before training the generator')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

n_words = 40
###### Definition of variables ######
# Networks
netG_A2B = Generator(n_words, temp=opt.temp)
netG_B2A = Generator(n_words, temp=opt.temp)
netD_A = Discriminator(n_words)
netD_B = Discriminator(n_words)
netC = CycleLoss(n_words)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if opt.epoch > 0:
    print(f"Attempting to start from epoch {opt.epoch}")
    checkpoint = torch.load(f'output/fullModel_{opt.epoch}.pth')
    netG_A2B.load_state_dict(checkpoint['genAB'])
    netG_B2A.load_state_dict(checkpoint['genBA'])
    netD_A.load_state_dict(checkpoint['discA'])
    netD_B.load_state_dict(checkpoint['discB'])
    netC.load_state_dict(checkpoint['cycleL'])

    checkpoint = torch.load(f'output/fullModelOptims_{opt.epoch}.pth')
    optimizer_G.load_state_dict(checkpoint['gen_o'])
    optimizer_D_A.load_state_dict(checkpoint['discA_o'])
    optimizer_D_B.load_state_dict(checkpoint['discB_o'])
    optimizer_cycle.load_state_dict(checkpoint['cycleL_o'])


netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)
netC.to(device)

# Lossess
criterion_GAN = torch.nn.CrossEntropyLoss()
criterion_cycle = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_cycle = torch.optim.Adam(netC.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_cycle = torch.optim.lr_scheduler.LambdaLR(optimizer_cycle, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
target_real = Variable(Tensor(opt.batchSize).fill_(1), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0), requires_grad=False)

data_stream = torch.load('data.pth')


results = {'GAN_AB': [], 'GAN_BA': [],
 'D_A': [], 'D_B': [],
 'C_A': [], 'C_B': [],
 'AA': []}
###################################


###### Training ######
for epoch in range(0, opt.n_epochs):
    run_loss_AB = 0
    run_loss_BA = 0
    run_loss_D_A = 0
    run_loss_D_B = 0

    run_loss_C_A = 0
    run_loss_C_B = 0
    run_loss_AA = 0

    for i, (nes, lakh) in enumerate(tqdm(data_stream)):


        real_A = nes.clone().detach().to(device)
        real_B = lakh.clone().detach().to(device)
        real_A = F.one_hot(real_A, 631).float()
        real_B = F.one_hot(real_B, 631).float()

        ###### Generators A2B and B2A ######
        if i % opt.gen_train == 0:
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B).long()
            loss_cycle_ABA = criterion_cycle(netC(real_A, recovered_A), target_real)

            recovered_B = netG_A2B(fake_A).long()
            loss_cycle_BAB = criterion_cycle(netC(real_B, recovered_B), target_real)

            # Total loss
            loss_G = ((loss_GAN_A2B + loss_GAN_B2A) + (loss_cycle_ABA + loss_cycle_BAB))*0.25

            loss_G.backward()
            optimizer_G.step()

            run_loss_AB += loss_GAN_A2B
            run_loss_BA += loss_GAN_B2A
            run_loss_C_A += loss_cycle_ABA
            run_loss_C_B += loss_cycle_BAB
        else:
            with torch.no_grad():
                # GAN loss
                fake_B = netG_A2B(real_A)
                fake_A = netG_B2A(real_B)


        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A.detach())
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        run_loss_D_A += loss_D_A

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B.detach())
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()


        run_loss_D_B += loss_D_B

        ###### Cycle loss ######
        optimizer_cycle.zero_grad()

        # A-A
        pred_real = netC(real_A.detach(), real_A.detach())
        loss_real_A = criterion_cycle(pred_real, target_real)

        # A-A*
        pred_fake = netC(real_A.detach(), fake_A.detach())
        loss_fake_A = criterion_cycle(pred_fake, target_fake)

        # B-B
        pred_real = netC(real_B.detach(), real_B.detach())
        loss_real_B = criterion_cycle(pred_real, target_real)

        # B-B*
        pred_fake = netC(real_B.detach(), fake_B.detach())
        loss_fake_B = criterion_cycle(pred_fake, target_fake)

        # Total loss
        loss_cycle = (loss_real_A + loss_real_B + loss_fake_A + loss_fake_B)*0.25
        loss_cycle.backward()

        optimizer_cycle.step()

        run_loss_AA += loss_cycle

        if i % 100 == 99:
            print(f"Batch/Epoch: {i}/{epoch}")
            print(f"GAN loss: {(run_loss_AB +  run_loss_BA) / i * opt.gen_train}")
            print(f"Autoencoder loss: {run_loss_AA / i}")
            print(f"Discriminator loss: {(run_loss_D_A +  run_loss_D_B) /i}")

        del real_A
        del real_B
        del fake_A
        del fake_B

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    lr_scheduler_cycle.step()

    results['GAN_AB'].append(run_loss_AB.item())
    results['GAN_BA'].append(run_loss_BA.item())
    results['D_A'].append(run_loss_D_A.item())
    results['D_B'].append(run_loss_D_B.item())
    results['C_A'].append(run_loss_C_A.item())
    results['C_B'].append(run_loss_C_B.item())
    results['AA'] .append(run_loss_AA.item())

    for key in results.keys():
        print(f'{key} loss: {results[key][-1]}')

    # Save models checkpoints
    torch.save(results, 'output/results.pth')
    # torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    # torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    # torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    # torch.save(netD_B.state_dict(), 'output/netD_B.pth')
    # torch.save(netC.state_dict(), 'output/netC.pth')

    save_checkpointFull(f'output/fullModel_{epoch}.pth',
                        genAB = netG_A2B,
                        genBA = netG_B2A,
                        discA = netD_A,
                        discB = netD_B,
                        cycleL = netC)

    save_checkpointFull(f'output/fullModelOptims_{epoch}.pth',
                    gen_o = optimizer_G,
                    discA_o = optimizer_D_A,
                    discB_o = optimizer_D_B,
                    cycleL_o = optimizer_cycle)

###################################
