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
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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


pretrain_dataset = PretrainingDataset(img_dir='data/CAM16_100cls_10mask/test/data/normal', transform=transform)

pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)

sizes = []
for i, image in tqdm(enumerate(pretrain_loader)):
    image_size = image.shape
    size = (image_size[1], image_size[2], image_size[3])
    if size not in sizes:
        sizes.append(size)
    


        
print(sizes)
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(data.shape)
#     print(target)
    
#     if batch_idx >5:
#         plt.imshow(data.squeeze().permute(1,2,0))
#         plt.savefig('test.png')
#         break