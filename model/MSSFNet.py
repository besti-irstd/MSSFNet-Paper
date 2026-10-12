"""MSSFNet: Mamba Spatial–Spectral Fusion Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import AWFE, FCSM, HWFP, MSDA, ResNet

class MSSFNet(nn.Module):
    """Mamba Spatial–Spectral Fusion Network."""

    def __init__(self, input_channels, block=ResNet):
        super(MSSFNet, self).__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        # ----------------------------------------
        # 1. 频域分支：多尺度特征提取
        # ----------------------------------------
        self.awfe = AWFE(input_channels)
        self.freq_e0 = nn.Sequential(
            nn.Conv2d(input_channels, param_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(param_channels[0]),
            nn.ReLU(inplace=True),
            block(param_channels[0], param_channels[0])
        )
        # HWFP 的三个阶段递归传递 LL 子带，输入通道数保持为 16。
        self.hwfp = HWFP(param_channels[0], param_channels[1:4])

        # ----------------------------------------
        # 2. 空间分支：编码器 (U-Net Encoder)
        # ----------------------------------------
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])

        # ----------------------------------------
        # 3. FCSM：频率条件化空间 Mamba
        # ----------------------------------------
        self.fcsm_e3 = FCSM(in_channels=param_channels[3])
        self.fcsm_d0 = FCSM(in_channels=param_channels[0])

        # ----------------------------------------
        # 4. 空间分支：中间层与解码器 (U-Net Middle & Decoder)
        # ----------------------------------------
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])

        self.decoder_3 = self._make_layer2(param_channels[3] + param_channels[4], param_channels[3], block,
                                           param_blocks[2])
        self.decoder_2 = self._make_layer2(param_channels[2] + param_channels[3], param_channels[2], block,
                                           param_blocks[1])
        self.decoder_1 = self._make_layer2(param_channels[1] + param_channels[2], param_channels[1], block,
                                           param_blocks[0])
        self.decoder_0 = self._make_layer2(param_channels[0] + param_channels[1], param_channels[0], block)

        # ----------------------------------------
        # 5. 辅助输出与最终输出
        # ----------------------------------------
        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final_output = nn.Conv2d(param_channels[0], 1, kernel_size=1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load current weights or original weights through a name-only mapping."""
        from utils.checkpoint import remap_state_dict

        return super().load_state_dict(
            remap_state_dict(state_dict), strict=strict, assign=assign
        )

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(MSDA(in_channels, out_channels, 3, 3, 3))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def _make_layer2(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, deep_supervision=True):
        # --- 1. 频域特征流提取 (获取多尺度频域特征) ---
        x_freq = self.awfe(x)
        f0 = self.freq_e0(x_freq)  # [B, 16, H, W]，充当初始的 LL 子带

        # HWFP 保留三个尺度的计算；当前融合位置使用 f0 和 f3。
        f1, f2, f3 = self.hwfp(f0)

        # --- 2. 空间特征流提取 (U-Net 编码器阶段) ---
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        # --- 3. 融合点 1: 深层跨域交互 ---
        x_e3_fused = self.fcsm_e3(x_e3, f3)

        # --- 4. 继续 U-Net 路径 (Middle + Decoder) ---
        x_m = self.middle_layer(self.pool(x_e3_fused))

        x_d3 = self.decoder_3(torch.cat([x_e3_fused, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        # --- 5. 融合点 2: 浅层边缘增强 ---
        feat_fused_final = self.fcsm_d0(x_d0, f0)

        # --- 6. 生成多尺度掩码与最终预测 ---
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)

        output = self.final_output(feat_fused_final)

        # 深监督掩码上采样到输入分辨率，与标签对齐。
        if deep_supervision:
            mask1 = F.interpolate(mask1, size=mask0.shape[2:], mode='bilinear', align_corners=True)
            mask2 = F.interpolate(mask2, size=mask0.shape[2:], mode='bilinear', align_corners=True)
            mask3 = F.interpolate(mask3, size=mask0.shape[2:], mode='bilinear', align_corners=True)
            return [mask0, mask1, mask2, mask3], output
        else:
            return [], output
