"""Shared convolution, residual block and Haar wavelet operators."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .csa import CSA

class CDC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 init_theta=0.7):
        super(CDC_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.theta = nn.Parameter(torch.full((1, out_channels, 1, 1), init_theta))

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        out_normal = self.conv(x)

        weight = self.conv.weight
        kernel_sum = weight.sum(dim=(2, 3), keepdim=True)
        mask = torch.zeros_like(weight)
        mask[:, :, weight.shape[2] // 2, weight.shape[3] // 2] = 1
        weight_diff = weight - kernel_sum * mask

        out_diff = F.conv2d(input=x, weight=weight_diff, bias=self.conv.bias,
                            stride=self.conv.stride, padding=self.conv.padding,
                            dilation=self.conv.dilation, groups=self.conv.groups)

        theta_clamped = self.theta

        return out_normal * (1 - theta_clamped) + out_diff * theta_clamped


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.csa = CSA(out_channels)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.csa(out)
        out += residual
        out = self.relu(out)
        return out


class HaarDWT(nn.Module):
    def __init__(self, channels):
        super(HaarDWT, self).__init__()
        self.channels = channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        filters = torch.stack([ll, hl, lh, hh], dim=0).unsqueeze(1)
        filters = filters.repeat(channels, 1, 1, 1)
        self.weight = nn.Parameter(filters, requires_grad=False)

    def forward(self, x):
        out = F.conv2d(x, self.weight, stride=2, groups=self.channels)
        return out


class HaarIDWT(nn.Module):
    def __init__(self, channels):
        super(HaarIDWT, self).__init__()
        self.channels = channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        filters = torch.stack([ll, hl, lh, hh], dim=0).unsqueeze(1)
        filters = filters.repeat(channels, 1, 1, 1)
        self.weight = nn.Parameter(filters, requires_grad=False)

    def forward(self, x):
        out = F.conv_transpose2d(x, self.weight, stride=2, groups=self.channels)
        return out
