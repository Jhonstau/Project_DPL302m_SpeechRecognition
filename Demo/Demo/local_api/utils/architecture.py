import torch.nn.functional as F
import torch.nn as nn
import torch

class Conv3D(nn.Module):
    def __init__(
        self,
        num_classes=8,
        conv_channels=[32, 64, 128, 128, 256],
        dropout=0.3,
        fc_dim=128,
        input_shape=(1, 2, 130, 40)
    ):
        super().__init__()

        self.conv_layers = nn.Sequential()
        in_channels = input_shape[0]

        for i, out_channels in enumerate(conv_channels):
            self.conv_layers.add_module(f"conv3d_{i}", nn.Conv3d(
                in_channels, out_channels, kernel_size=(3, 3, 3), padding=1
            ))
            self.conv_layers.add_module(f"gn_{i}", nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels))
            self.conv_layers.add_module(f"relu_{i}", nn.ReLU())
            self.conv_layers.add_module(f"dropout3d_{i}", nn.Dropout3d(p=0.1))
            self.conv_layers.add_module(f"pool_{i}", nn.MaxPool3d(kernel_size=(1, 2, 2)))
            in_channels = out_channels

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Tính output dim
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy_out = self.global_pool(self.conv_layers(dummy))
            self.flatten_dim = dummy_out.view(1, -1).shape[1]

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.flatten_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)