"""AWFE: Adaptive Wavelet Frequency Enhancement."""

import torch
import torch.nn as nn

from .basic import HaarDWT, HaarIDWT

class AWFE(nn.Module):
    def __init__(self, in_channels):
        super(AWFE, self).__init__()
        self.dwt = HaarDWT(in_channels)
        self.idwt = HaarIDWT(in_channels)

        self.alpha = nn.Parameter(torch.tensor(0.1))

        high_freq_channels = 3 * in_channels
        mid_channels = max(3, high_freq_channels // 2)
        self.high_freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_freq_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, high_freq_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        dwt_out = self.dwt(x)
        dwt_out = dwt_out.view(B, C, 4, dwt_out.shape[2], dwt_out.shape[3])

        LL = dwt_out[:, :, 0, :, :]
        high_freqs = dwt_out[:, :, 1:, :, :]

        LL_filtered = LL * self.alpha
        high_freqs_reshaped = high_freqs.contiguous().view(B, 3 * C, high_freqs.shape[3], high_freqs.shape[4])
        attn_weights = self.high_freq_attn(high_freqs_reshaped)
        high_freqs_weighted = high_freqs_reshaped * attn_weights
        high_freqs_weighted = high_freqs_weighted.view(B, C, 3, dwt_out.shape[3], dwt_out.shape[4])

        filtered_dwt = torch.cat([LL_filtered.unsqueeze(2), high_freqs_weighted], dim=2)
        filtered_dwt = filtered_dwt.view(B, 4 * C, dwt_out.shape[3], dwt_out.shape[4])

        out = self.idwt(filtered_dwt)
        return out
