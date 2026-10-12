"""HWFP: Hierarchical Wavelet Frequency Pyramid with three recursive stages."""

import torch.nn as nn

from .basic import HaarDWT

class HWFPStage(nn.Module):
    """
    接收上一级的低频子带 (LL)，进行真正的 DWT 分解。
    输出当前尺度的频域融合特征 (f1/f2/f3)，同时输出纯净的下一级 LL 子带。
    """

    def __init__(self, in_channels, out_channels):
        super(HWFPStage, self).__init__()
        self.dwt = HaarDWT(in_channels)
        # 将 DWT 产生的 4 个子带（LL, HL, LH, HH）拼接后映射到目标通道数
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 小波变换 (获取 4*C 个通道)
        dwt_out = self.dwt(x)  # shape: [B, 4*C, H/2, W/2]

        # 2. 分离 LL 子带供下一级使用
        # HaarDWT 的输出通道排列为：每个输入通道对应 4 个输出通道 (LL, HL, LH, HH)
        dwt_reshaped = dwt_out.view(B, C, 4, dwt_out.shape[2], dwt_out.shape[3])
        LL_next = dwt_reshaped[:, :, 0, :, :].contiguous()  # 提取第 0 个分量即为 LL

        # 3. 融合所有频域分量作为当前尺度跨域融合的控制信号
        feat_out = self.fuse(dwt_out)  # shape: [B, out_channels, H/2, W/2]

        return feat_out, LL_next


class HWFP(nn.Module):
    """Build three scales by passing each stage's LL band to the next stage."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if len(out_channels) != 3:
            raise ValueError("HWFP requires exactly three output channel counts")
        self.stage1 = HWFPStage(in_channels, out_channels[0])
        self.stage2 = HWFPStage(in_channels, out_channels[1])
        self.stage3 = HWFPStage(in_channels, out_channels[2])

    def forward(self, x):
        f1, ll_1 = self.stage1(x)
        f2, ll_2 = self.stage2(ll_1)
        f3, _ = self.stage3(ll_2)
        return f1, f2, f3
