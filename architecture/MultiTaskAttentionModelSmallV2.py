import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskAttentionModelSmallV2(nn.Module):
    def __init__(self, num_countries, num_denoms, embed_dim=192, num_heads=6, dropout=0.25):
        super(MultiTaskAttentionModelSmallV2, self).__init__()

        # CNN Backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.conv4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(embed_dim)
        self.conv5 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(embed_dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 196, embed_dim))  # 14x14 patches

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Pool + heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc_country = nn.Linear(embed_dim, num_countries)
        self.fc_denom = nn.Linear(embed_dim, num_denoms)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, N, C)

        # Positional embedding
        pos_emb = (
            F.interpolate(self.pos_embedding.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False)
            .transpose(1, 2)
        )
        x = x + pos_emb[:, :x.size(1), :]

        # Transformer
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)

        return self.fc_country(x), self.fc_denom(x)

