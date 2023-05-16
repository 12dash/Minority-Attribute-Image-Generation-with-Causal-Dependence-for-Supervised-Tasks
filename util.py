import os
import torch
import numpy as np

import matplotlib.pyplot as plt

def make_dirs(path):
    try :
        os.makedirs(path)
    except Exception as e:
        pass

def plot_image(orig, fake_img, epoch, num_imgs = 10, plot_dir = 'plot/'):
    _, ax = plt.subplots(2, num_imgs, figsize=(24,4))
    with torch.no_grad():
        orig = np.transpose(orig.cpu().numpy(), (0, 2, 3, 1))
        for i in range(num_imgs):
            ax[0][i].imshow(orig[i])
            ax[0][i].set_xticks([])
            ax[0][i].set_yticks([])

        fake_img = np.transpose(fake_img.cpu().numpy(), (0, 2, 3, 1))
        for i in range(num_imgs):
            ax[1][i].imshow(fake_img[i])
            ax[1][i].set_xticks([])
            ax[1][i].set_yticks([])
    plt.savefig(f"{plot_dir}{epoch}.jpg")

def save_model(bgm, disc, epoch, model_dir = 'saved_model/'):
    make_dirs(f"saved_model/{model_dir}")
    
    bgm_path = f'saved_model/{model_dir}bgm'
    torch.save({'epoch': epoch, 'model_state_dict': bgm.state_dict()}, bgm_path)

    disc_path = f'saved_model/{model_dir}disc'
    torch.save({'epoch': epoch, 'model_state_dict': disc.state_dict()}, disc_path)