import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# data pre-processing
train_set = transforms.Compose ([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

val_set = transforms.Compose ([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

class FairFace(Dataset):
    def __init__(self, df, root_dir, transform= None):
        self.df = df.reset_index (drop = True)
        self.root_dir = "dataset"
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_rel_path = self.df.loc[idx, 'file']
        img_path = os.path.join(self.root_dir, img_rel_path)

        image = Image.open(img_path).convert('RGB')

        label = self.df.loc[idx, 'gender']
        label = 1 if label == 'Male' else 0

        if self.transform:
            image = self.transform(image)
        
        return image, label
    



