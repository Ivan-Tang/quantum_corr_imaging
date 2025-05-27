import torch
import torch.nn as nn
import torch.nn.functional as F

class CompressiveImagingModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256, n_heads=4, n_layers=2, output_size=512*384):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.attn_pool = nn.Sequential(
            nn.Linear(input_dim, 1),  
            nn.Softmax(dim=1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid() 
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1) 

        encoded = self.encoder(x)

        attn_weights = self.attn_pool(encoded)
        pooled = torch.sum(attn_weights * encoded, dim=1)

        out = self.decoder(pooled)
        return out.view(-1, 1, 512, 384)