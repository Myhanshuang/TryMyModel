import torch
from torch import nn


class Patches(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv2d(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class VisionEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_t = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x_t + self.dropout1(attn_out)
        x_t = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_t + x
        return x


class ViTEncoder(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1, output_dim=None):
        super().__init__()

        self.patch_embedding = Patches(img_size, patch_size, in_channels, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.n_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            VisionEncoderBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        final_output_dim = output_dim
        self.projection = nn.Linear(d_model, final_output_dim)

    def forward(self, x):
        x = self.patch_embedding(x)
        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # -> [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # -> [B, N+1, D]

        x = x + self.pos_embedding[:, :(N + 1)]
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        cls_output = x[:, 0]

        final_output = self.projection(cls_output)

        return final_output