import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Keep the hidden channel count positive for small inputs.
        hidden_planes = max(in_planes // ratio, 4)

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Multiscale spatial attention (MS-SAM)
class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(MultiScaleSpatialAttention, self).__init__()

        # Local 3x3 convolution for fine stroke details.
        self.conv_local = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

        # Dilated 3x3 convolution with a 5x5 receptive field.
        self.conv_context = nn.Conv2d(2, 1, kernel_size=3, padding=2, dilation=2, bias=False)

        # Fuse local and contextual responses.
        self.conv_fuse = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute channel-wise average and maximum projections.
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_compress = torch.cat([avg_out, max_out], dim=1)  # Shape: [B, 2, H, W]

        # Extract spatial features at two scales.
        out_local = self.conv_local(x_compress)  # Local features
        out_context = self.conv_context(x_compress)  # Contextual features

        # Fuse both branches using a 1x1 convolution.
        out_cat = torch.cat([out_local, out_context], dim=1)
        out = self.conv_fuse(out_cat)

        return self.sigmoid(out)


# CBAM with multiscale spatial attention
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        # Apply multiscale spatial attention.
        self.sa = MultiScaleSpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# Convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Attention-enhanced U-Net
class EnhancedUNet(nn.Module):
    """
    U-Net with channel attention and multiscale spatial attention.
    Used for stroke-width map estimation.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.down1 = ConvBlock(in_channels, 64)
        self.down2 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder and attention blocks
        self.up_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Attention is applied after skip-connection fusion.
        self.up1 = ConvBlock(256, 128)
        self.att1 = CBAM(128)  # Attention block

        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = ConvBlock(128, 64)
        self.att2 = CBAM(64)  # Attention block

        # Output projection
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encode input features.
        x1 = self.down1(x)
        p1 = self.pool(x1)
        x2 = self.down2(p1)
        p2 = self.pool(x2)

        # Process bottleneck features.
        b = self.bottleneck(p2)

        # First decoder stage
        u1 = self.up_conv1(b)
        cat1 = torch.cat([u1, x2], dim=1)
        x3 = self.up1(cat1)
        x3 = self.att1(x3)  # Apply attention.

        # Second decoder stage
        u2 = self.up_conv2(x3)
        cat2 = torch.cat([u2, x1], dim=1)
        x4 = self.up2(cat2)
        x4 = self.att2(x4)  # Apply attention.

        return self.out_conv(x4)