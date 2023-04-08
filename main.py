import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from Encoder import Encoder, reparameterize
from Generator import Generator
from Discriminator import BigJointDiscriminator

from utils import get_train_dataloader

def save_image(fake, epoch = None):
    with torch.no_grad():
        fake = np.transpose(fake.cpu().numpy(), (0, 2, 3, 1))
    _,ax = plt.subplots(1, 10, figsize=(24,4))
    for i in range(10):
        ax[i].imshow(fake[i])
    name = "final"
    if epoch is not None:
        name = str(epoch)
    plt.savefig(f'plots/{name}.png')

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    root_folder = 'sample_data/'

    in_channels = 3
    fc_size = 2048
    latent_dim = 100

    img_dim = 64
    batch_size = 32

    train_dataloader = get_train_dataloader(root_folder, img_dim=img_dim, batch_size=batch_size)

    e = Encoder(latent_dim = latent_dim, fc_size = fc_size).to(device)
    g = Generator(latent_dim = latent_dim, image_size = img_dim).to(device)
    disc = BigJointDiscriminator(latent_dim = latent_dim, image_size = img_dim).to(device)
 
    e_optimizer = optim.Adam(e.parameters(), lr=5e-5, betas=(0, 0.999))
    g_optimizer = optim.Adam(g.parameters(), lr=5e-5, betas=(0, 0.999))
    D_optimizer = optim.Adam(disc.parameters(), lr=1e-4, betas=(0, 0.999))
    
    epochs = 1

    num = len(train_dataloader.dataset)//batch_size
    print('Begining to train')

    for epoch in range(epochs):
        disc_loss = []
        e_loss = []
        g_loss = []
        
        e.train()
        g.train()
        disc.train()

        for (X, y) in train_dataloader:
            disc.zero_grad()
            
            X = X.to(device)
            z = torch.randn(X.shape[0], latent_dim, device=device)
            mu, sigma = e(X)
            z_fake = reparameterize(mu, sigma )
            X_fake = g(z)
            
            e_score = disc(X, z_fake.detach())
            g_score = disc(X_fake.detach(), z.detach())
            
            del z_fake
            del X_fake
            
            loss_d = F.softplus(g_score).mean() + F.softplus(-e_score).mean()
            loss_d.backward()
            D_optimizer.step()
            disc_loss.append(loss_d.item())
            #___________________________________
            e.zero_grad()
            g.zero_grad()
            
            mu, sigma = e(X)
            z_fake = reparameterize(mu, sigma )
            X_fake = g(z)
            
            e_score = disc(X, z_fake)
            l_encoder = e_score.mean()
            l_encoder.backward()
            e_loss.append(l_encoder.item())
            e_optimizer.step()
            
            g_score = disc(X_fake, z)
            s_decoder = torch.exp(g_score.detach()).clamp(0.5, 2)
            loss_decoder = -(s_decoder * g_score).mean()
            g_loss.append(loss_decoder.item())
            loss_decoder.backward()
            g_optimizer.step()
            
        print(f"[{epoch+1}/{epochs}] Encoder Loss : {sum(e_loss)/num:>.5f} Gen Loss : {sum(g_loss)/num:>.5f} Disc Loss : {sum(disc_loss)/num:>.5f}")
        if (epoch+1) % 5 == 0 or epoch == 0:
            e.eval()
            g.eval()
            for X, y in train_dataloader:
                mu, sigma = e(X.to(device))
                z = reparameterize(mu, sigma)
                x_fake = g(z)
                x_fake = (x_fake * 0.5) + 0.5
                save_image(x_fake, epoch+1)
                break