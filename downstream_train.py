import torch
from torch import nn
import torch.optim as optim

from models.resnet import *
from load_data import *
from util import *
import time
from sklearn.metrics import accuracy_score

class ImageDataset(Dataset):
    def __init__(self, df, transform, attr, label = None):
        self.transform=transform
        self.df = df.replace(-1,0)
        self.image_path = self.df.pop('path')
        self.attr = self.df[attr].values
        self.label = self.df[label].values
   
    def __len__(self):
        return len(self.image_path)
 
    def __getitem__(self, index):
        image_path = self.image_path[index]
        image=Image.open(image_path)
        image=self.transform(image)
        label = torch.tensor(self.label[index], dtype=torch.float32)
        return image, label, torch.tensor(self.attr[index], dtype=torch.float32)

class Model(nn.Module):
    def __init__(self, in_dim, out_dim = 1):
        super().__init__()
        self.base_model = resnet18(pretrained=False, in_channels=in_dim, fc_size=2048, out_dim=out_dim).to(device)
    
    def forward(self, x):
        return self.base_model(x)

def train_step(model, dataloader, optimizer, criterion):
    running_loss = []
    for i, data in enumerate(train_dataloader):
        inputs, labels, attr = data
        inputs, labels, attr = inputs.to(device), labels.to(device), attr.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    return np.mean(running_loss)

global device

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    in_dim = 3
    out_dim = 1
    img_dim = 64
    batch_size = 128

    epochs = 15

    model = Model(in_dim = in_dim, out_dim = out_dim).to(device)
    transform = Compose([CenterCrop(128),
                         Resize((img_dim, img_dim)),
                         ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    folder = "downsample_smile_lm_100"

    data_dir = f'synthetic_dataset/{folder}/'
    file_name = 'df.csv'         
    df = pd.read_csv(f"{data_dir}{file_name}").reset_index(drop=True)
    attr = 'Male'
    label = 'Smiling'  
    data = ImageDataset(df = df, attr = attr, transform = transform, label = label)
    train_dataloader = DataLoader(data, batch_size = batch_size,shuffle = True)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):  
        t1 = time.time()
        running_loss = train_step(model, train_dataloader, optimizer, criterion)
        train_time = (time.time() - t1) / 60
        print(f'[{epoch + 1}] loss: {running_loss:.3f} Time:{train_time:.3f}')
    print('Finished Training')

    result_folder = "results/"
    make_dirs(result_folder)
    test_df = pd.read_csv(f"dataset/celebA/dear_test.csv")
    test_df['path'] = test_df['image_id'].apply(lambda x: 'dataset/celebA/img/img_align_celeba/'+x)
    test_datal = ImageDataset(df = test_df, attr = attr, transform = transform, label = label)
    test_dataloader = DataLoader(test_datal, batch_size = batch_size, shuffle = True)
    
    correct_pred = {0:{0:0,1:0},1:{0:0,1:0}}
    total_pred = {0:{0:0,1:0},1:{0:0,1:0}}
    model.eval()
    label_l, pred_l = [], []

    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels, attrs = data
            inputs, labels, attrs = inputs.to(device), labels.to(device), attrs.to(device)
            outputs, _ = model(inputs)
            predictions = torch.round(torch.sigmoid(outputs))
            for label, prediction, attr in zip(labels, predictions, attrs):
                if label == prediction:
                    correct_pred[label.item()][attr.item()] += 1
                total_pred[label.item()][attr.item()] += 1
                label_l.append(label.item())
                pred_l.append(prediction.item())
    
    result_file = f"{result_folder}{folder}.txt"

    with open(result_file,'w') as file:
        for classname, correct_counts in correct_pred.items():
            for attr_name, correct_count in correct_counts.items():
                accuracy = 100 * float(correct_count) / total_pred[classname][attr_name]
                s = (f'Accuracy for class: {classname} , attr: {attr_name}: {accuracy} total: {total_pred[classname][attr_name]}\n')
                file.write(s)
        
        acc = accuracy_score(label_l, pred_l)
        acc_s = f"\n Accuracy Overall : {acc}"
        file.write(acc_s)



