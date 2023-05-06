import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, CenterCrop

class ImageDataset(Dataset):
    def __init__(self, root_folder, file_name, transform, cols = None):
        self.transform=transform
        self.img_folder=root_folder+'img/img_align_celeba/'
        
        self.attr = pd.read_csv(root_folder+file_name+'.csv').sample(frac=0.005).replace(-1,0).reset_index(drop=True)
        self.image_names = self.attr.pop('image_id')
        if cols is not None:
            self.attr = self.attr[cols]    
        self.num_feat = len(self.attr.columns)
        self.order = list(self.attr.columns)
        
        self.attr = self.attr.values
   
    def __len__(self):
        return len(self.image_names)
 
    def __getitem__(self, index):
        image_path = self.img_folder + self.image_names[index]
        image=Image.open(image_path)
        image=self.transform(image)
        label = torch.tensor(self.attr[index], dtype = torch.float)
        return image, label

def get_dataloader(root_folder, file_name = 'dear_train', img_dim=64, batch_size=32, cols = None):
    transform = Compose([CenterCrop(128),
                        Resize((img_dim, img_dim)),
                        ToTensor(),
                        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = ImageDataset(root_folder=root_folder, file_name = file_name, transform=transform, cols = cols)
    dataloader = DataLoader(data, batch_size = batch_size, num_workers = 2, 
                                  shuffle = True, prefetch_factor = 4)
    return dataloader