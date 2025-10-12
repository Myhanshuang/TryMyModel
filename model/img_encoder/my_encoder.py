import torch
from torch import nn


class MyVisionEncoder(nn.Module):
    def __init__(self, output_dim=256, in_channel=3, dropout=0.4):
        super().__init__()
        self.block1_out_channel = 32
        self.block2_out_channel = 128

        self.block1_num = 2
        self.block2_num = 2

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=self.block1_out_channel,out_channels=self.block1_out_channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.block1_out_channel),
            nn.ReLU()
        )
        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=self.block1_out_channel, kernel_size=1),
            nn.BatchNorm2d(self.block1_out_channel),
            nn.ReLU()
        )
        self.block1 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.block1_out_channel, out_channels=self.block1_out_channel, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.block1_out_channel),
            nn.ReLU()
        ) for _ in range(self.block1_num)])
        self.block1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1_block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.block1_out_channel, out_channels=self.block2_out_channel, kernel_size=1),
            nn.BatchNorm2d(self.block2_out_channel),
            nn.ReLU()
        )
        self.block2 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.block2_out_channel, out_channels=self.block2_out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.block2_out_channel),
            nn.ReLU(),
        ) for _ in range(self.block2_num)])
        self.block2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=self.block1_out_channel,out_channels=self.block2_out_channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.block2_out_channel),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.block2_out_channel, output_dim)

    def forward(self, x):
        x = self.block0(x)
        x_t = x
        for layer in self.block1:
            x = layer(x)
        x = self.block1_maxpool(x)
        x = x + self.shortcut1(x_t)

        x_t = x
        x = self.block1_block2(x)
        for layer in self.block2:
            x = layer(x)
        x = self.block2_maxpool(x)
        x = x + self.shortcut2(x_t)
        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.linear(self.dropout(x))
        return x