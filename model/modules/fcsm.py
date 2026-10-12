"""FCSM: Frequency-Conditioned Spatial Mamba with four scan directions."""

import warnings

import torch
import torch.nn as nn

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    warnings.warn(
        "FCSM requires mamba-ssm and causal-conv1d with compatible CUDA support "
        "for forward execution.",
        ImportWarning,
        stacklevel=2,
    )
    selective_scan_fn = None

class FCSM(nn.Module):
    def __init__(self, in_channels, d_state=16):
        super(FCSM, self).__init__()
        self.d_model = in_channels
        self.d_state = d_state
        self.dt_rank = max(1, in_channels // 16)

        self.in_proj_s = nn.Conv2d(in_channels, in_channels, 1)

        self.ctrl_dim = self.dt_rank + self.d_state * 2
        self.x_proj_f = nn.Conv2d(in_channels, self.ctrl_dim * 4, 1)

        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, in_channels) for _ in range(4)
        ])

        self.A_logs = nn.ParameterList([
            nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float).repeat(in_channels, 1)))
            for _ in range(4)
        ])
        self.Ds = nn.ParameterList([
            nn.Parameter(torch.ones(in_channels)) for _ in range(4)
        ])

        self.out_norm = nn.LayerNorm(in_channels)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, feat_spatial, feat_freq):
        B, C, H, W = feat_spatial.shape
        L = H * W

        xs = self.in_proj_s(feat_spatial)
        xf_params = self.x_proj_f(feat_freq)

        xs_1 = xs.view(B, C, L).transpose(1, 2)
        xs_2 = torch.flip(xs.view(B, C, L), dims=[2]).transpose(1, 2)
        xs_3 = xs.transpose(2, 3).contiguous().view(B, C, L).transpose(1, 2)
        xs_4 = torch.flip(xs.transpose(2, 3).contiguous().view(B, C, L), dims=[2]).transpose(1, 2)
        xs_dirs = [xs_1, xs_2, xs_3, xs_4]

        xf_1 = xf_params.view(B, -1, L).transpose(1, 2)
        xf_2 = torch.flip(xf_params.view(B, -1, L), dims=[2]).transpose(1, 2)
        xf_3 = xf_params.transpose(2, 3).contiguous().view(B, -1, L).transpose(1, 2)
        xf_4 = torch.flip(xf_params.transpose(2, 3).contiguous().view(B, -1, L), dims=[2]).transpose(1, 2)
        xf_dirs = [xf_1, xf_2, xf_3, xf_4]

        y_out = 0

        for i in range(4):
            u = xs_dirs[i]
            ctrl = xf_dirs[i]

            ctrl_i = ctrl[:, :, i * self.ctrl_dim: (i + 1) * self.ctrl_dim]
            dt, B_mat, C_mat = torch.split(ctrl_i, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            dt = self.dt_projs[i](dt)
            A = -torch.exp(self.A_logs[i])
            D = self.Ds[i]

            u_scan = u.transpose(1, 2).contiguous()
            dt_scan = dt.transpose(1, 2).contiguous()
            B_scan = B_mat.transpose(1, 2).contiguous()
            C_scan = C_mat.transpose(1, 2).contiguous()

            if selective_scan_fn is None:
                raise RuntimeError("FCSM 需要安装支持 CUDA 的 mamba-ssm 和 causal-conv1d。")

            y_i = selective_scan_fn(
                u_scan,
                dt_scan,
                A,
                B_scan,
                C_scan,
                D,
                z=None,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False
            )

            y_i = y_i.transpose(1, 2)

            if i == 0:
                y_out = y_out + y_i.transpose(1, 2).view(B, C, H, W)
            elif i == 1:
                y_out = y_out + torch.flip(y_i.transpose(1, 2), dims=[2]).view(B, C, H, W)
            elif i == 2:
                y_out = y_out + y_i.transpose(1, 2).view(B, C, W, H).transpose(2, 3)
            elif i == 3:
                y_out = y_out + torch.flip(y_i.transpose(1, 2), dims=[2]).view(B, C, W, H).transpose(2, 3)

        y_out = y_out.permute(0, 2, 3, 1)
        y_out = self.out_norm(y_out).permute(0, 3, 1, 2)

        out = self.out_proj(y_out)
        return out
