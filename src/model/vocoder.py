import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torchinfo import summary


class ResBlock(nn.Module):
    """
    Residual Block for HiFiGAN generator
    """

    def __init__(self, hid_channels, kernel_size, dilations, relu_slope=0.1):
        super().__init__()
        M, L = len(dilations), len(dilations[0])
        self.layers = nn.ModuleList([])
        for m in range(M):
            for l in range(L):  # noqa
                self.layers.append(
                    nn.Sequential(
                        nn.LeakyReLU(relu_slope),
                        weight_norm(
                            nn.Conv1d(
                                in_channels=hid_channels,
                                out_channels=hid_channels,
                                kernel_size=kernel_size,
                                dilation=dilations[m][l],
                                padding="same",
                            )
                        ),
                    )
                )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = x + layer(x)
        return x


class MRF(nn.Module):
    """
    MRF block for HiFiGAN generator
    """

    def __init__(self, kernels, dilations, hid_channels=512, relu_slope=0.1):
        super().__init__()
        assert len(kernels) == len(
            dilations
        ), f"Kernels and dilations must be same size ({len(kernels)} != {len(dilations)})"

        N = len(dilations)
        self.res_blocks = nn.ModuleList([])
        for n in range(N):
            self.res_blocks.append(
                ResBlock(
                    hid_channels=hid_channels,
                    kernel_size=kernels[n],
                    dilations=dilations[n],
                    relu_slope=relu_slope,
                )
            )

    def forward(self, x: torch.Tensor):
        out = 0.0
        for res_block in self.res_blocks:
            out = out + res_block(x)

        return out


class Generator(nn.Module):
    """
    HiFiGAN generator
    """

    def __init__(
        self,
        in_channels=80,
        hid_channels=512,
        kernels=[16, 16, 4, 4],
        mrf_kernels=[3, 7, 11],
        mrf_dilations=[[[1, 1], [3, 1], [5, 1]]] * 3,
        relu_slope=0.1,
    ):
        """
        Args:
            in_channels [int]: input mel spec channels
            hid_channels [int]: hidden channels
            kernels [list]: kernel sizes for transpose convs in each layer
            mrf_kernels [list]: kernels of MRF in res blocks
            mrf_dilations [list]: dilations of MRF in res blocks
            relu_slope [float]: negative slope for LeakyReLU
        """
        super().__init__()

        self.init_conv = weight_norm(
            nn.Conv1d(in_channels=in_channels, out_channels=hid_channels, kernel_size=7)
        )

        self.layers = nn.ModuleList([])
        in_ch = hid_channels
        num_layers = len(kernels)
        for i in range(num_layers):
            out_ch = in_ch // 2
            self.layers.append(
                nn.Sequential(
                    weight_norm(
                        nn.ConvTranspose1d(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=kernels[i],
                            stride=kernels[i] // 2,
                        )
                    ),
                    MRF(mrf_kernels, mrf_dilations, out_ch, relu_slope),
                )
            )
            in_ch = in_ch // 2

        self.final_conv = nn.Sequential(
            nn.LeakyReLU(relu_slope),
            weight_norm(nn.Conv1d(in_channels=in_ch, out_channels=1, kernel_size=7)),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        # gate
        x = self.init_conv(x)

        # main layers
        for layer in self.layers:
            x = layer(x)
            print(x.shape)

        # gate to audio
        x = self.final_conv(x)

        return x


class PeriodDiscr(nn.Module):
    def __init__(self, in_channels=1, period=3, num_layers=4, relu_slope=0.1):
        super().__init__()
        self.period = period

        self.layers = nn.ModuleList([])
        in_ch = in_channels
        for i in range(1, num_layers + 1):
            out_ch = 2 ** (5 + i)
            self.layers.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            # TODO: add padding
                        )
                    ),
                    nn.LeakyReLU(relu_slope),
                )
            )
            in_ch = out_ch

        self.final_conv = nn.Sequential(
            weight_norm(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=1024,
                    kernel_size=(5, 1),
                    padding="same",
                )
            ),
            nn.LeakyReLU(relu_slope),
            weight_norm(
                nn.Conv2d(
                    in_channels=1024, out_channels=1, kernel_size=(3, 1), padding="same"
                )
            ),
        )

    def forward(self, x: torch.Tensor):
        x = self._pad_and_reshape(x)

        for layer in self.layers:
            x = layer(x)

        out = torch.flatten(self.final_conv(x), 1, -1)
        return out

    def _pad_and_reshape(self, x: torch.Tensor):
        # [B, 1, T] -> [B, 1, ceil(T / p), p]
        B, C, T = x.shape
        padding = self.period - (T % self.period)
        x = F.pad(x, (0, padding))
        x = x.reshape(B, C, x.shape[-1] // self.period, self.period)
        return x


class MPD(nn.Module):
    def __init__(
        self, in_channels=1, periods=[2, 3, 5, 7, 11], num_layers=4, relu_slope=0.1
    ):
        super().__init__()
        self.discrs = nn.ModuleList([])
        for p in periods:
            self.discrs.append(PeriodDiscr(in_channels, p, num_layers, relu_slope))

    def forward(self, x: torch.Tensor):
        out = []
        for discr in self.discrs:
            out.append(discr(x))
        return out


class SubDiscr(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hid_channels=128,
        kernel_size=41,
        groups=16,
        strides=[2, 4, 4],
        pooling=1,
        is_first=False,
    ):
        super().__init__()
        """
        norm_f(Conv1d(1, 128, 15, 1, padding=7)),
        norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),

        norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
        norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
        norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),

        norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
        norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        """
        self.pooling = pooling
        norm = spectral_norm if is_first else weight_norm

        self.init_gate = nn.Sequential(
            norm(nn.Conv1d(in_channels, hid_channels, 15, padding=7)),
            norm(
                nn.Conv1d(
                    hid_channels,
                    hid_channels,
                    kernel_size,
                    groups=4,
                    padding=kernel_size // 2,
                )
            ),
        )

        self.layers = nn.ModuleList([])
        N = len(strides)
        for i in range(N):
            self.layers.append(
                norm(
                    nn.Conv1d(
                        hid_channels,
                        2 * hid_channels,
                        kernel_size,
                        stride=strides[i],
                        groups=groups,
                        padding=kernel_size // 2,
                    )
                )
            )
            hid_channels = hid_channels * 2

        self.final_conv = nn.Sequential(
            norm(
                nn.Conv1d(
                    hid_channels,
                    hid_channels,
                    kernel_size,
                    groups=groups,
                    padding=kernel_size // 2,
                )
            ),
            norm(nn.Conv1d(hid_channels, hid_channels, 5, padding=2)),
            norm(nn.Conv1d(hid_channels, 1, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor):
        x = F.avg_pool1d(x, kernel_size=self.pooling, padding=self.pooling // 2)
        x = self.init_gate(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_conv(x)
        x = torch.flatten(x, 1, -1)
        return x


class MSD(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hid_channels=128,
        kernel_size=41,
        groups=16,
        strides=[2, 4, 4],
        pools=[1, 2, 4],
    ):
        super().__init__()
        self.discrs = nn.ModuleList([])
        N = len(pools)
        for i in range(N):
            self.discrs.append(
                SubDiscr(
                    in_channels,
                    hid_channels,
                    kernel_size,
                    groups,
                    strides,
                    pooling=pools[i],
                    is_first=(i == 0),
                )
            )

    def forward(self, x: torch.Tensor):
        out = []
        for discr in self.discrs:
            out.append(discr(x))
        return out
