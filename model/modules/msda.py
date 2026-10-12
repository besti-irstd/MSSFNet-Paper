"""MSDA: Multi-Scale Differential Aggregation."""

import torch
import torch.nn as nn

from .basic import CDC_Conv

class MSDA(nn.Module):
    def __init__(self, inplanes, outplanes, one, two, three, scales=4):
        super(MSDA, self).__init__()
        if outplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        self.scales = scales
        self.relu = nn.ReLU(inplace=True)
        self.spx = outplanes // scales
        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, 1, 0),
            nn.BatchNorm2d(outplanes)
        )
        self.conv1 = nn.Sequential(
            CDC_Conv(self.spx, self.spx, kernel_size=one, stride=1, padding=one // 2, groups=self.spx),
            nn.BatchNorm2d(self.spx),
        )
        self.conv2 = nn.Sequential(
            CDC_Conv(self.spx, self.spx, kernel_size=two, stride=1, padding=2, dilation=2, groups=self.spx),
            nn.BatchNorm2d(self.spx),
        )
        self.conv3 = nn.Sequential(
            CDC_Conv(self.spx, self.spx, kernel_size=three, stride=1, padding=1, groups=self.spx),
        )
        self.conv4 = nn.Sequential(
            CDC_Conv(self.spx, self.spx, kernel_size=three, stride=1, padding=2, dilation=2, groups=self.spx),
        )
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.spx)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.inconv(x)
        inputt = x
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(self.relu(self.conv1(xs[1])))
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))
        temp = xs[3] + ys[2]
        temp1 = self.conv5(self.conv3(temp) + self.conv4(temp))
        ys.append(self.relu(temp1))
        y = torch.cat(ys, 1)

        y = self.outconv(y)
        output = self.relu(y + inputt)
        return output
