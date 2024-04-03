import torch
import torch.nn as nn
import torch.nn.functional as F


class Merge_And_Run_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Merge_And_Run_Unit, self).__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True)
        )
        self.merge = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        r1 = self.path1(x)
        r2 = self.path2(x)
        out = torch.cat([r1, r2], dim=1)
        c_out = self.merge(out)
        out = c_out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.block(x)
        out = out + x
        return F.relu(out, inplace=True)


class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out = self.block(x) + x
        return F.relu(out, inplace=True)


class LocalConnection(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(LocalConnection, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.block(x)
        out = out * x
        return out


class EAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EAM, self).__init__()
        self.block1 = Merge_And_Run_Unit(in_channels, out_channels)
        self.block2 = ResidualBlock(out_channels, out_channels)
        self.block3 = EnhancedResidualBlock(out_channels, out_channels)
        self.LC = LocalConnection(out_channels)

    def forward(self, x):
        r1 = self.block1(x)
        r2 = self.block2(r1)
        r3 = self.block3(r2)
        out = self.LC(r3)
        out = out + x
        return out


class RIDNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_features=64, *args, **kwargs):
        super(RIDNet, self).__init__()
        self.feature_extraction = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)

        self.eam1 = EAM(num_features, num_features)
        self.eam2 = EAM(num_features, num_features)
        self.eam3 = EAM(num_features, num_features)
        self.eam4 = EAM(num_features, num_features)

        self.conv = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.feature_extraction(x)
        r1 = self.eam1(x1)
        r2 = self.eam2(r1)
        r3 = self.eam3(r2)
        r4 = self.eam4(r3)

        x2 = x1 + r4  # Long skip Connection
        x3 = self.conv(x2)
        out = x + x3  # Long skip Connection
        return out
