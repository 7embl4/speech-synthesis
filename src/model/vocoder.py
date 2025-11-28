import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torch.nn.utils.rnn import pad_sequence


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

        return {"gen_audio": x}


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

        self.final_gates = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            in_channels=in_ch,
                            out_channels=1024,
                            kernel_size=(5, 1),
                            padding="same",
                        )
                    ),
                    nn.LeakyReLU(relu_slope),
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=1024,
                        out_channels=1,
                        kernel_size=(3, 1),
                        padding="same",
                    )
                ),
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self._pad_and_reshape(x)

        fmap = []
        for layer in self.layers:
            x = layer(x)
            fmap.append(x)

        for gate in self.final_gates:
            x = gate(x)
            fmap.append(x)

        x = torch.flatten(x, 1, -1)
        return x, fmap

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

    def forward(self, real_audio: torch.Tensor, gen_audio: torch.Tensor):
        rs, gs, r_fmaps, g_fmaps = [], [], [], []
        for discr in self.discrs:
            r, r_fmap = discr(real_audio)
            g, g_fmap = discr(gen_audio)
            rs.append(r)
            gs.append(g)
            r_fmaps.append(r_fmap)
            g_fmaps.append(g_fmap)

        return rs, gs, r_fmaps, g_fmaps


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
        relu_slope=0.1,
    ):
        super().__init__()
        self.pooling = pooling
        norm = spectral_norm if is_first else weight_norm

        self.init_gates = nn.ModuleList(
            [
                nn.Sequential(
                    norm(nn.Conv1d(in_channels, hid_channels, 15, padding=7)),
                    nn.LeakyReLU(relu_slope),
                ),
                nn.Sequential(
                    norm(
                        nn.Conv1d(
                            hid_channels,
                            hid_channels,
                            kernel_size,
                            groups=4,
                            padding=kernel_size // 2,
                        )
                    ),
                    nn.LeakyReLU(relu_slope),
                ),
            ]
        )

        self.layers = nn.ModuleList([])
        N = len(strides)
        for i in range(N):
            self.layers.append(
                nn.Sequential(
                    norm(
                        nn.Conv1d(
                            hid_channels,
                            2 * hid_channels,
                            kernel_size,
                            stride=strides[i],
                            groups=groups,
                            padding=kernel_size // 2,
                        )
                    ),
                    nn.LeakyReLU(relu_slope),
                )
            )
            hid_channels = hid_channels * 2

        self.final_gates = nn.ModuleList(
            [
                nn.Sequential(
                    norm(
                        nn.Conv1d(
                            hid_channels,
                            hid_channels,
                            kernel_size,
                            groups=groups,
                            padding=kernel_size // 2,
                        )
                    ),
                    nn.LeakyReLU(relu_slope),
                ),
                nn.Sequential(
                    norm(nn.Conv1d(hid_channels, hid_channels, 5, padding=2)),
                    nn.LeakyReLU(relu_slope),
                ),
                norm(nn.Conv1d(hid_channels, 1, 3, padding=1)),
            ]
        )

    def forward(self, x: torch.Tensor):
        x = F.avg_pool1d(x, kernel_size=self.pooling, padding=self.pooling // 2)

        fmap = []
        for gate in self.init_gates:
            x = gate(x)
            fmap.append(x)

        for layer in self.layers:
            x = layer(x)
            fmap.append(x)

        for gate in self.final_gates:
            x = gate(x)
            fmap.append(x)

        x = torch.flatten(x, 1, -1)
        return x, fmap


class MSD(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hid_channels=128,
        kernel_size=41,
        groups=16,
        strides=[2, 4, 4],
        pools=[1, 2, 4],
        relu_slope=0.1,
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
                    relu_slope=relu_slope,
                )
            )

    def forward(self, real_audio: torch.Tensor, gen_audio: torch.Tensor):
        rs, gs, r_fmaps, g_fmaps = [], [], [], []
        for discr in self.discrs:
            r, r_fmap = discr(real_audio)
            g, g_fmap = discr(gen_audio)
            rs.append(r)
            gs.append(g)
            r_fmaps.append(r_fmap)
            g_fmaps.append(g_fmap)

        return rs, gs, r_fmaps, g_fmaps


class Discriminator(nn.Module):
    def __init__(
        self,
        # mpd
        mpd_periods=[2, 3, 5, 7, 11],
        mpd_num_layers=4,
        # msd
        msd_hid_channels=128,
        msd_kernel_size=41,
        msd_groups=16,
        msd_strides=[2, 4, 4],
        msd_pools=[1, 2, 4],
        # common
        audio_channels=1,
        relu_slope=0.1,
    ):
        super().__init__()
        self.mpd = MPD(audio_channels, mpd_periods, mpd_num_layers, relu_slope)
        self.msd = MSD(
            audio_channels,
            msd_hid_channels,
            msd_kernel_size,
            msd_groups,
            msd_strides,
            msd_pools,
            relu_slope,
        )

    def forward(self, real_audio: torch.Tensor, gen_audio: torch.Tensor):
        real_audio, gen_audio = self._pad_to_equal(real_audio, gen_audio)

        mpd_rs, mpd_gs, mpd_r_fmaps, mpd_g_fmaps = self.mpd(real_audio, gen_audio)
        msd_rs, msd_gs, msd_r_fmaps, msd_g_fmaps = self.msd(real_audio, gen_audio)

        return {
            "mpd_rs": mpd_rs,
            "mpd_gs": mpd_gs,
            "mpd_r_fmaps": mpd_r_fmaps,
            "mpd_g_fmaps": mpd_g_fmaps,
            "msd_rs": msd_rs,
            "msd_gs": msd_gs,
            "msd_r_fmaps": msd_r_fmaps,
            "msd_g_fmaps": msd_g_fmaps,
        }

    def _pad_to_equal(self, real_audio: torch.Tensor, gen_audio: torch.Tensor):
        diff = abs(real_audio.shape[-1] - gen_audio.shape[-1])

        if real_audio.shape[-1] < gen_audio.shape[-1]:
            real_audio = F.pad(real_audio, (0, diff), mode="reflect")
        else:
            gen_audio = F.pad(gen_audio, (0, diff), mode="reflect")

        return real_audio, gen_audio


class HiFiGAN(nn.Module):
    def __init__(
        self,
        # generator
        mel_channels=80,
        hid_channels=512,
        gen_kernels=[16, 16, 4, 4],
        mrf_kernels=[3, 7, 11],
        mrf_dilations=[[[1, 1], [3, 1], [5, 1]]] * 3,
        # discriminator
        mpd_periods=[2, 3, 5, 7, 11],
        mpd_num_layers=4,
        msd_hid_channels=128,
        msd_kernel_size=41,
        msd_groups=16,
        msd_strides=[2, 4, 4],
        msd_pools=[1, 2, 4],
        # common
        audio_channels=1,
        relu_slope=0.1,
    ):
        super().__init__()
        self.generator = Generator(
            mel_channels,
            hid_channels,
            gen_kernels,
            mrf_kernels,
            mrf_dilations,
            relu_slope,
        )
        self.discriminator = Discriminator(
            mpd_periods,
            mpd_num_layers,
            msd_hid_channels,
            msd_kernel_size,
            msd_groups,
            msd_strides,
            msd_pools,
            audio_channels,
            relu_slope,
        )

    def forward(self, spec, **batch):
        return self.generator(spec)

    def generate(self, spec, **batch):
        return self.generator(spec)

    def discriminate(self, audio, gen_audio, **batch):
        return self.discriminator(audio, gen_audio)
