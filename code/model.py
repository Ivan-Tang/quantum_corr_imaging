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

        # 卷积解码器：先升维成特征图，再用转置卷积还原空间结构
        self.fc = nn.Linear(input_dim, 128 * 16 * 12)  # 128通道，16x12特征图
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64x48
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 128x96
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # 256x192
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),     # 512x384
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1) 

        encoded = self.encoder(x)

        attn_weights = self.attn_pool(encoded)
        pooled = torch.sum(attn_weights * encoded, dim=1)  # [B, input_dim]

        feat = self.fc(pooled)  # [B, 128*16*12]
        feat = feat.view(-1, 128, 16, 12)  # [B, 128, 16, 12]
        out = self.decoder(feat)  # [B, 1, 512, 384]
        return out