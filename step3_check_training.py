import torch
from models.mae import MAE
from models.vit import ViT
from models.conv_vit_autoencoder import ConvViTAutoencoder
from torchvision import transforms
from step1_analyze_data import PretrainingDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def denormalize_img(img, mean, std):
    for t,m,s in zip(img, mean, std):
        t.mul_(s).add_(m)
        
    img = torch.clamp(img, 0,1)
    return img.permute((1,2,0))

if __name__ == '__main__':
    device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'  
    print('using device ', device)
    
    conv_vit_ae = ConvViTAutoencoder(
        image_size = 384,
        patch_size = 16,
        num_classes = 5,
        latent_dim = 512, # final 
        dropout = 0.2,
        encoder_depth = 6,
        encoder_heads = 8,
        encoder_mlp_dim = 2048,
        encoder_dim_head = 64,
        decoder_depth = 6,
        decoder_heads = 8,
        decoder_mlp_dim = 2048,
        decoder_dim_head = 64
    )
    conv_vit_ae = conv_vit_ae.to(device)
    
    
    conv_vit_ae.load_state_dict(torch.load('checkpoints/conv_vit_ae_1.pt'))
    
    mean= [0.6618, 0.5137, 0.6184]
    std=[0.1878, 0.2276, 0.1912]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    pretrain_dataset = PretrainingDataset(img_dir='data/CAM16_100cls_10mask/train/data/normal', transform=transform)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)


    for step, image_batch in tqdm(enumerate(pretrain_loader)):
        plt.figure()
        image_batch = image_batch.to(device)

        patches = conv_vit_ae.to_patch(image_batch)
        _, patch_encoding = conv_vit_ae.encode(patches)
        
        r_image_batch = conv_vit_ae.to_img(conv_vit_ae.decode(patch_encoding))
        
        img = image_batch.detach().cpu().squeeze()
        
        r_img = r_image_batch.detach().cpu().squeeze()

        plt.subplot(121)
        plt.title('img')
        plt.imshow(denormalize_img(img, mean, std))
        plt.subplot(122)
        plt.title('recon')
        plt.imshow(denormalize_img(r_img, mean, std))
        plt.savefig(f'output/{step}.png')
        
        if step > 3:
            break
        
    
    