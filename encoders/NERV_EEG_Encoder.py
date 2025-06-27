import torch.nn as nn

class NERVEncoder(nn.Module):
    def __init__(self, n_channels, in_samples, embed_dim=1024):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_channels*in_samples, embed_dim))
        # Two 2D conv branches on (channels x time):
        #  - Spatial-then-temporal (S→T)
        self.conv_st = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(n_channels,1)),  # spatial conv across channels
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, 3), padding=(0,1)),  # temporal conv
            nn.ReLU(),
        )
        #  - Temporal-then-spatial (T→S)
        self.conv_ts = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,3), padding=(0,1)),  # temporal conv
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(n_channels,1)),       # spatial conv
            nn.ReLU(),
        )
        # Transformer-style attention over time and channels
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        # Final projection to embedding
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x: Tensor [batch, channels, time]
        B, C, T = x.size()
        # Prepare for conv: add dummy channel dim
        x_unsq = x.unsqueeze(1)  # [B, 1, C, T]
        out_st = self.conv_st(x_unsq)  # [B, 16, 1, T] -> squeeze channel dim
        out_ts = self.conv_ts(x_unsq)  # [B, 16, 1, T]
        # Combine conv features
        conv_feat = (out_st + out_ts).squeeze(2)  # [B, 16, T]
        # Flatten time-channel and add position encoding
        feat = conv_feat.flatten(start_dim=1)  # [B, 16*T]
        feat = feat.unsqueeze(0)  # [1, B, 16*T] for attention
        # Self-attention (learns inter-channel/time relations)
        attn_out, _ = self.attn(feat + self.pos_embed, feat + self.pos_embed, feat + self.pos_embed)
        attn_out = attn_out.squeeze(0)  # [B, 16*T]
        # Project to final EEG embedding
        eeg_emb = self.fc(attn_out)     # [B, embed_dim]
        return eeg_emb  # e.g. 1024-dim embedding
