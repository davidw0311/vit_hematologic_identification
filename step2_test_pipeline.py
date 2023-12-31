import torch
from models.mae import MAE
from models.vit import ViT
from models.conv_vit_autoencoder import ConvViTAutoencoder
from torchvision import transforms
from step1_analyze_data import PretrainingDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



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
    
    print('total number of parameters: ', sum(p.numel() for p in conv_vit_ae.parameters() if p.requires_grad))
    
    
    mean= [0.6618, 0.5137, 0.6184]
    std=[0.1878, 0.2276, 0.1912]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    pretrain_dataset = PretrainingDataset(img_dir='data/CAM16_100cls_10mask/train/data/normal', transform=transform)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=8, shuffle=False)

    # sizes = []
    
    num_epochs = 100
    optimizer = torch.optim.Adam(conv_vit_ae.parameters(), lr=0.001)
    for e in range(1, num_epochs+1):
        print( f"{f'starting epoch {e}':-^{50}}" )
        for step, image_batch in tqdm(enumerate(pretrain_loader)):
            image_batch = image_batch.to(device)
            optimizer.zero_grad()
            loss = conv_vit_ae(image_batch)
            loss.backward()
            optimizer.step()
            
            if step%10 == 0 :
                print(f'loss after step {step+1}: ',loss.detach().item())
        
        if e%5 == 0 or e == 1:
            torch.save(conv_vit_ae.state_dict(), f'checkpoints/conv_vit_ae_{e}.pt')
