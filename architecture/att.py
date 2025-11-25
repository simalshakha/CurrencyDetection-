
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskAttentionModelSmall(nn.Module):
    def __init__(self, num_countries, num_denoms, embed_dim=128, num_heads=4, dropout=0.2):
        super(MultiTaskAttentionModelSmall, self).__init__()

        # --- CNN Backbone ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.conv4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(embed_dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 196, embed_dim))  # 14x14 patches

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Pooling + dropout
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Heads
        self.fc_country = nn.Linear(embed_dim, num_countries)
        self.fc_denom = nn.Linear(embed_dim, num_denoms)

    def forward(self, x):
        # CNN backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # (B, embed_dim, H, W)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, N, C)

        # Positional embedding
        if x.size(1) > self.pos_embedding.size(1):
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=x.size(1),
                mode="linear",
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_emb = self.pos_embedding[:, :x.size(1), :]
        x = x + pos_emb

        # Transformer
        x = self.transformer(x)  # (B, N, C)

        # Pooling + dropout
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)

        # Heads
        country_logits = self.fc_country(x)
        denom_logits = self.fc_denom(x)

        return country_logits, denom_logits
