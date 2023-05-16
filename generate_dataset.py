import torch
from models.bgm import *
from models.sagan import *
from models.causal_model import *

import pandas as pd
from util import *
from load_data import *

class ImageDataset(Dataset):
    def __init__(self, root_folder, df, transform, cols = None):
        self.transform=transform
        self.img_folder=root_folder+'img/img_align_celeba/'
        self.attr = df.copy()
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
        attr = torch.tensor(self.attr[index], dtype = torch.float)
        return image, attr, index, image_path

def get_dataloader(root_folder, df, img_dim=64, batch_size=32, cols = None):
    transform = Compose([Resize((img_dim, img_dim)),
                        ToTensor(),
                        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = ImageDataset(root_folder=root_folder, df = df, transform=transform, cols = cols)
    dataloader = DataLoader(data, batch_size = batch_size, shuffle = True)
    return dataloader

def save_fig(img_batch, img_id, folder):
    img_paths_ = []
    for idx, img in enumerate(img_batch):
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
        img_path = folder+str(img_id[idx])+"_t.jpg"
        img.save(img_path)
        img_paths_.append(img_path)
    return img_paths_

def get_model(model_name, num_label, A, latent_dim = 100):
    in_channels = 3
    fc_size = 2048
    

    g_conv_dim = 32
    enc_dist='gaussian'
    enc_arch='resnet'
    enc_fc_size=2048
    enc_noise_dim=128
    dec_dist = 'implicit'
    prior = 'linscm'

    model = BGM(latent_dim, g_conv_dim, img_dim,
                enc_dist, enc_arch, enc_fc_size, enc_noise_dim, dec_dist,
                prior, num_label, A)

    model = nn.DataParallel(model)
    checkpoint = torch.load(f'saved_model/{model_name}/bgm', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module.to(device)
    return model

def convert_img(model, dc_loader, attr_new , y_new, img_path_, label_, attr_, make_sample, param):
    with torch.no_grad():
        for x, label, index, _ in dc_loader:
            x = x.to(device)
            #eps = model.encode(x)
            eps = torch.randn(x.size(0), latent_dim, device=device)
            label_z = model.prior(eps[:, :num_label])
            orig_z = label_z.detach()
            label_z[:, idx_attr] = param['min_attr'] if attr_new == 0 else param['max_attr']
            label_z[:, idx_y] = param['min_label'] if y_new == 0 else param['max_label']
            label_z = model.prior.intervene(label_z, orig_z)
            other_z = eps[:, num_label:]
            z = torch.cat([label_z, other_z], dim=1)
            new_img = model.decoder(z)
            new_img = np.transpose(new_img.cpu().numpy(), (0, 2, 3, 1))
            new_img = (new_img*0.5 + 0.5) * 255
            prev_img_id = list(df_temp.iloc[index]['image_id'])
            prev_img_id = [x.replace(".jpg","") for x in prev_img_id]
            paths = save_fig(new_img, prev_img_id, dest_img_folder)
            
            img_path_ = img_path_ + paths
            attr_ = attr_ + [attr_new for _ in range(len(prev_img_id))]
            label_ = label_ + [y_new for _ in range(len(prev_img_id))]
            
            if len(label_) > make_sample:
                break
    return img_path_, label_, attr_

global device

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    root_folder = "dataset/celebA/"
    img_folder = f"{root_folder}img/img_align_celeba/"
    
    img_dim = 64
    cols = ['Smiling', 'Male', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Narrow_Eyes', 'Chubby']
    num_label = len(cols)

    ## Defining the Adjacency Matrix ##
    A = torch.zeros((num_label, num_label))
    A[0, 2:6] = 1
    A[1, 4] = 1

    ## Choose Model ##
    #model_name = 'entiredata'
    #latent_dim = 100
    #range_param = {
    #  'min_attr':-5, 'max_attr' : 5,
    #    'min_label':-5, 'max_label':5
    #}

    #
    #latent_dim = 10
    #model_name = f'downsample_smile_lm_{latent_dim}_'
    #range_param = {
    #    'min_attr':-20, 'max_attr' : 4,
    #    'min_label':-20, 'max_label':4
    #}

    latent_dim = 100
    range_param = {
        'min_attr':-20, 'max_attr' : 4,
        'min_label':-20, 'max_label':4
    }

    model_name = f'downsample_smile_lm_{latent_dim}'
    
    
    #latent_dim = 150
    #range_param = {
    #    'min_attr':-25, 'max_attr' : 25,
    #    'min_label':-10, 'max_label':10
    #}
    #model_name = f'downsample_smile_lm_{latent_dim}'
    
    model = get_model(model_name, num_label, A, latent_dim)

    ### Define Destiination ###
    dest_dir = f"synthetic_dataset/{model_name}/"
    dest_img_folder = dest_dir+"img/"
    make_dirs(dest_img_folder)

    ## Training Smaple ##
    file_name = "dear_train_downsample_smile"
    df = pd.read_csv(f"dataset/celebA/{file_name}.csv").replace(-1,0)

    y_name, attr_name = 'Smiling', 'Male'
    idx_attr = cols.index(attr_name)
    idx_y = cols.index(y_name)

    batch_size = 128
    make_sample = 5000
    
    img_path_, label_, attr_ = [], [], []

    attr_new , y_new = 0, 0
    df_temp = df[(df['Smiling']==0) & (df['Male']==1)].reset_index(drop=True)
    dc1_loader = get_dataloader(root_folder, df_temp, batch_size = batch_size)
    img_path_, label_, attr_  = convert_img(model, dc1_loader, attr_new , y_new, img_path_, label_, attr_, make_sample, range_param)
    print('1st Conversion Done')


    df_temp = df[(df['Smiling']==1) & (df['Male']==0)].reset_index(drop=True)
    dc2_loader = get_dataloader(root_folder, df_temp, batch_size = batch_size)
    attr_new , y_new = 1, 1
    img_path_, label_, attr_  = convert_img(model, dc2_loader, attr_new , y_new, img_path_, label_, attr_, 2*make_sample, range_param)
    print('2nd Conversion Done')

    for idx,row in df.iterrows():
        image_id = row['image_id']
        attr_val = row[attr_name]
        y_val = row[y_name]
        src_ = f'{img_folder}{image_id}'

        img_path_.append(src_) 
        label_.append(y_val) 
        attr_.append(attr_val)

    n = make_sample

    new_df = pd.DataFrame.from_dict({'path':img_path_, f'{y_name}': label_ ,f'{attr_name}':attr_})
    df1 = new_df[(new_df[y_name] == 1) & (new_df[attr_name] == 1)].reset_index(drop=True).sample(n=n+1000)
    df2 = new_df[(new_df[y_name] == 1) & (new_df[attr_name] == 0)].reset_index(drop=True).sample(n=n)
    df3 = new_df[(new_df[y_name] == 0) & (new_df[attr_name] == 0)].reset_index(drop=True).sample(n=n)
    df4 = new_df[(new_df[y_name] == 0) & (new_df[attr_name] == 1)].reset_index(drop=True).sample(n=n)
    new_df = pd.concat([df1,df2,df3, df4])
    new_df.to_csv(dest_dir+"df.csv", index=False)

