import torch
import numpy as np

import matplotlib.pyplot as plt

def plot_image(img, epoch, num_imgs = 10, plot_dir = 'plot/'):
    _, ax = plt.subplots(1, num_imgs, figsize=(24,4))
    with torch.no_grad():
        img = np.transpose(img.cpu().numpy(), (0, 2, 3, 1))
        for i in range(num_imgs):
            ax[i].imshow(fake[i])
    plt.savefig(f"{plot_dir}{epoch}.jpg")

def save_model(model_name, model, epoch, model_dir = 'saved_model/'):
    path = f'{model_dir}{model_name}'
    dic = {'epoch': epoch, 'model_state_dict': model.state_dict()}
    torch.save(dic, path)