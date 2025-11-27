import torch
from torch import nn
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
                        nn.Conv1d(
                            in_channels=hid_channels,
                            out_channels=hid_channels,
                            kernel_size=kernel_size,
                            dilation=dilations[m][l],
                            padding="same",
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

        self.init_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=hid_channels, kernel_size=7
        )

        self.layers = nn.ModuleList([])
        in_ch = hid_channels
        num_layers = len(kernels)
        for i in range(num_layers):
            out_ch = in_ch // 2
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernels[i],
                        stride=kernels[i] // 2,
                    ),
                    MRF(mrf_kernels, mrf_dilations, out_ch, relu_slope),
                )
            )
            in_ch = in_ch // 2

        self.final_conv = nn.Sequential(
            nn.LeakyReLU(relu_slope),
            nn.Conv1d(in_channels=in_ch, out_channels=1, kernel_size=7),
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


if __name__ == "__main__":
    gen = Generator(80, 128, [16, 16, 4, 4], [3, 7, 11], [[[1, 1], [3, 1], [5, 1]]] * 3)
    print(summary(gen))
    # x = torch.rand(1, 80, 862)
    # out = gen(x)
    # print(out.shape)
