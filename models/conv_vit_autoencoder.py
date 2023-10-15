# code referenced and modified from https://github.com/lucidrains/vit-pytorch/tree/main

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ConvViTAutoencoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, latent_dim, channels = 3, dropout = 0.,
                 encoder_depth, encoder_heads, encoder_mlp_dim, encoder_dim_head, 
                 decoder_depth, decoder_heads, decoder_mlp_dim, decoder_dim_head):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.to_img = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_height // patch_height, w=image_width // patch_width, p1 = patch_height, p2 = patch_width)
                                    
        self.embed_patches = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.encoder_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, latent_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.encoder = Transformer(dim=latent_dim, depth=encoder_depth, heads=encoder_heads, dim_head=encoder_dim_head, mlp_dim=encoder_mlp_dim, dropout=dropout)

        self.decoder_pos_embedding = nn.Parameter(torch.randn(1, num_patches, latent_dim))
        self.decoder = Transformer(dim=latent_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, mlp_dim=decoder_mlp_dim, dropout=dropout)
        self.to_pixels = nn.Linear(latent_dim, patch_dim)


    def encode(self, patches):
        
        batch_size, num_patches, _ = patches.shape

        tokens = self.embed_patches(patches)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
        tokens_with_cls = torch.cat((cls_tokens, tokens), dim=1)
        
        tokens_with_pos_embedding = tokens_with_cls + self.encoder_pos_embedding

        encoded_tokens = self.encoder(tokens_with_pos_embedding)
        
        cls_encoding = encoded_tokens[:, 0] # use for prediction later
        
        patch_encoding = encoded_tokens[:, 1:]
        
        return cls_encoding, patch_encoding
    
    def decode(self, patch_encoding):
        decoder_encoding = patch_encoding + self.decoder_pos_embedding
        decoded_tokens = self.decoder(decoder_encoding)
        
        reconstructed_patches = self.to_pixels(decoded_tokens)
        return reconstructed_patches
    

    def forward(self, img):
        patches = self.to_patch(img)
        cls_encoding, patch_encoding = self.encode(patches)
        reconstructed_patches = self.decode(patch_encoding)
        
        loss = F.mse_loss(reconstructed_patches, patches)
            
        return loss
        