from bgm import *
from sagan import *
from causal_model import *
from util import plot_image, save_model
from load_data import *

import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional as F

global device
global celoss

def train_step(train_dataloader, model, discriminator, A_optimizer, 
                prior_optimizer, encoder_optimizer, decoder_optimizer, disc_optimizer, 
                d_steps_per_iter = 1, g_steps_per_iter = 1, alpha = 5):
    model.train()
    discriminator.train()
    disc_loss, enc_loss, label_loss, gen_loss = [], [], [], []
    for batch_idx, (x, label) in enumerate(train_dataloader):
        x = x.to(device)
        sup_flag = label[:, 0] != -1
        if sup_flag.sum() > 0:
            label = label[sup_flag, :].float().to(device)

        for _ in range(d_steps_per_iter):
            discriminator.zero_grad()
            z = torch.randn(x.size(0), latent_dim, device=x.device)
            z_fake, x_fake, z, _ = model(x, z)
            encoder_score = discriminator(x, z_fake.detach())
            decoder_score = discriminator(x_fake.detach(), z.detach())

            del z_fake
            del x_fake

            loss_d = F.softplus(decoder_score).mean() + F.softplus(-encoder_score).mean()
            loss_d.backward()
            disc_optimizer.step()
            disc_loss.append(loss_d.item())
        
        for _ in range(g_steps_per_iter):
            #### ENCODER ####
            z = torch.randn(x.size(0), latent_dim, device=x.device)
            z_fake, x_fake, z, z_fake_mean = model(x, z)
            model.zero_grad()
            encoder_score = discriminator(x, z_fake)
            loss_encoder = encoder_score.mean()
            if sup_flag.sum() > 0:
                label_z = z_fake_mean[sup_flag, :num_label]
                sup_loss = celoss(label_z, label)
                label_loss.append(sup_loss.item())
            else:
                sup_loss = torch.zeros([1], device=device)
            loss_encoder = loss_encoder + sup_loss * alpha
            loss_encoder.backward()
            encoder_optimizer.step()
            prior_optimizer.step()
            enc_loss.append(loss_encoder.item())

            model.zero_grad()

            #### DECODER ####
            z = torch.randn(x.size(0), latent_dim, device=x.device)
            z_fake, x_fake, z, z_fake_mean = model(x, z)
            decoder_score = discriminator(x_fake, z)
            r_decoder = torch.exp(decoder_score.detach())
            s_decoder = r_decoder.clamp(0.5, 2)
            loss_decoder = -(s_decoder * decoder_score).mean()
            loss_decoder.backward()
            decoder_optimizer.step()
            model.module.prior.set_zero_grad()
            A_optimizer.step()
            prior_optimizer.step()
            gen_loss.append(loss_decoder.item())
    
    return np.mean(enc_loss), np.mean(gen_loss), np.mean(disc_loss), np.mean(label_loss)

def eval_step(dataloader, model, discriminator, epoch, save=True, num_imgs=10, alpha = 5):
    model.eval()
    discriminator.eval()
    disc_loss, enc_loss, label_loss, gen_loss = [], [], [], []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(dataloader):
            x = x.to(device)
            sup_flag = label[:, 0] != -1
            if sup_flag.sum() > 0:
                label = label[sup_flag, :].float().to(device)

            ### Discriminator ###
            z = torch.randn(x.size(0), latent_dim, device=x.device)
            z_fake, x_fake, z, _ = model(x, z)
            encoder_score = discriminator(x, z_fake.detach())
            decoder_score = discriminator(x_fake.detach(), z.detach())

            del z_fake
            del x_fake

            loss_d = F.softplus(decoder_score).mean() + F.softplus(-encoder_score).mean()
            disc_loss.append(loss_d.item())

            ### Encoder ###
            z = torch.randn(x.size(0), latent_dim, device=x.device)
            z_fake, x_fake, z, z_fake_mean = model(x, z)
            encoder_score = discriminator(x, z_fake)
            loss_encoder = encoder_score.mean()
            if sup_flag.sum() > 0:
                label_z = z_fake_mean[sup_flag, :num_label]
                sup_loss = celoss(label_z, label)
                label_loss.append(sup_loss.item())
            else:
                sup_loss = torch.zeros([1], device=device)
            loss_encoder = loss_encoder + sup_loss * alpha
            enc_loss.append(loss_encoder.item())

            ### Decoder ###
            z = torch.randn(x.size(0), latent_dim, device=x.device)
            z_fake, x_fake, z, z_fake_mean = model(x, z)
            decoder_score = discriminator(x_fake, z)
            r_decoder = torch.exp(decoder_score.detach())
            s_decoder = r_decoder.clamp(0.5, 2)
            loss_decoder = -(s_decoder * decoder_score).mean()
            gen_loss.append(loss_decoder.item())

        for batch_idx, (x, label) in enumerate(dataloader):
            x = x.to(device)[:num_imgs]
            x_recon = model(x, recon=True)
            x_recon = (x_recon * 0.5) + 0.5
            plot_image(x_recon, epoch)
            if save:
                save_model(model, discriminator, epoch)
            break

    return np.mean(enc_loss), np.mean(gen_loss), np.mean(disc_loss), np.mean(label_loss)

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    celoss = torch.nn.BCEWithLogitsLoss()
    cols = ['Smiling', 'Male', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Narrow_Eyes', 'Chubby']
    num_label = len(cols)
    root_folder = 'dataset/celebA/'

    in_channels = 3
    fc_size = 2048
    latent_dim = 100

    img_dim = 64
    batch_size = 128

    ### BGM PARAMETERS ###
    g_conv_dim = 32
    enc_dist='gaussian'
    enc_arch='resnet'
    enc_fc_size=2048
    enc_noise_dim=128
    dec_dist = 'implicit'
    prior = 'linscm'

    ### DISCRIIMINATOR ###
    d_conv_dim = 32
    dis_fc_size = 1024

    ### Loading Data ###
    train_dataloader = get_dataloader(root_folder,'dear_train', img_dim=img_dim, 
                                    batch_size=batch_size, cols = cols)
    val_dataloader = get_dataloader(root_folder,'dear_val', img_dim=img_dim, 
                                    batch_size=batch_size, cols = cols)       
    test_dataloader = get_dataloader(root_folder,'dear_test', img_dim=img_dim, 
                                    batch_size=batch_size, cols = cols)                                    

    ### MATRIX ENCODING CAUSAL DIAGRAM ###
    A = torch.zeros((num_label, num_label), device = device)
    A[0, 2:6] = 1
    A[1, 4] = 1

    ### Instantiate Models ###
    model = BGM(latent_dim, g_conv_dim, img_dim,
                enc_dist, enc_arch, enc_fc_size, enc_noise_dim, dec_dist,
                prior, num_label, A)
    discriminator = BigJointDiscriminator(latent_dim, d_conv_dim, img_dim, dis_fc_size)

    ### Optimisers ###
    enc_param = model.encoder.parameters()
    dec_param = list(model.decoder.parameters())
    prior_param = list(model.prior.parameters())

    A_optimizer = optim.Adam(prior_param[0:1], lr=5e-4)
    prior_optimizer = optim.Adam(prior_param[1:], lr=5e-4, betas=(0, 0.999))
    encoder_optimizer = optim.Adam(enc_param, lr=5e-5, betas=(0, 0.999))
    decoder_optimizer = optim.Adam(dec_param, lr=5e-5, betas=(0, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0, 0.999))

    model = nn.DataParallel(model.to(device))
    discriminator = nn.DataParallel(discriminator.to(device))
    epochs = 50

    for epoch in range(epochs):
        # Train Step
        enc_loss, gen_loss, disc_loss, label_loss = train_step(train_dataloader, model, discriminator, A_optimizer, 
                prior_optimizer, encoder_optimizer, decoder_optimizer, disc_optimizer, 
                d_steps_per_iter = 1, g_steps_per_iter = 1, alpha = 5)
        print(f"[{epoch+1}/{epochs}] Enc Loss : {enc_loss:>.5f} Gen Loss : {gen_loss:>.5f} Disc Loss : {disc_loss:>.5f}  Label Loss : {label_loss:>.5f}")
        
        # Val Step
        if (epoch+1) % 5 == 0:
            enc_loss, gen_loss, disc_loss, label_loss = eval_step(val_dataloader, model, discriminator, epoch, num_imgs = 10)
            print(f"[VAL] Enc Loss : {enc_loss:>.5f} Gen Loss : {gen_loss:>.5f} Disc Loss : {disc_loss:>.5f}  Label Loss : {label_loss:>.5f} \n")

    enc_loss, gen_loss, disc_loss, label_loss = eval_step(test_dataloader, model, discriminator, 'Test', num_imgs = 10, save = False)
    print(f"[TEST] Enc Loss : {enc_loss:>.5f} Gen Loss : {gen_loss:>.5f} Disc Loss : {disc_loss:>.5f}  Label Loss : {label_loss:>.5f} \n")