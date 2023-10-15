import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
])

class PretrainingDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_list = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_list[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

if __name__ == "__main__":
    # pretrain_dataset = PretrainingDataset(img_dir='data/WBC_100/data/Basophil', transform=transform)
    pretrain_dataset = PretrainingDataset(img_dir='data/CAM16_100cls_10mask/train/data/normal', transform=transform)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for img in tqdm(pretrain_loader):
        channels_sum += torch.mean(img, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(img ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    print("mean: ", mean)
    print("std: ", std)



    


    

