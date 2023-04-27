import torch
import numpy as np

import matplotlib.pyplot as plt

def plot_image(img, epoch, num_imgs = 10, plot_dir = 'plot/'):
    _, ax = plt.subplots(1, num_imgs, figsize=(24,4))
    with torch.no_grad():
        img = np.transpose(img.cpu().numpy(), (0, 2, 3, 1))
        for i in range(num_imgs):
            ax[i].imshow(img[i])
    plt.savefig(f"{plot_dir}{epoch}.jpg")

def save_model(bgm, disc, epoch, model_dir = 'saved_model/'):
    bgm_path = f'{model_dir}bgm'
    torch.save({'epoch': epoch, 'model_state_dict': bgm.state_dict()}, bgm_path)

    disc_path = f'{model_dir}disc'
    torch.save({'epoch': epoch, 'model_state_dict': disc.state_dict()}, disc_path)