import torch
import numpy as np

import matplotlib.pyplot as plt

def plot_image(orig, fake_img, epoch, num_imgs = 10, plot_dir = 'plot/'):
    _, ax = plt.subplots(2, num_imgs, figsize=(24,4))
    with torch.no_grad():
        orig = np.transpose(orig.cpu().numpy(), (0, 2, 3, 1))
        for i in range(num_imgs):
            ax[0][i].imshow(orig[i])
            ax[0][i].xticks([])
            ax[0][i].yticks([])

        fake_img = np.transpose(fake_img.cpu().numpy(), (0, 2, 3, 1))
        for i in range(num_imgs):
            ax[1][i].imshow(fake_img[i])
            ax[1][i].xticks([])
            ax[1][i].yticks([])
    plt.savefig(f"{plot_dir}{epoch}.jpg")

def save_model(bgm, disc, epoch, model_dir = 'saved_model/'):
    bgm_path = f'{model_dir}bgm'
    torch.save({'epoch': epoch, 'model_state_dict': bgm.state_dict()}, bgm_path)

    disc_path = f'{model_dir}disc'
    torch.save({'epoch': epoch, 'model_state_dict': disc.state_dict()}, disc_path)