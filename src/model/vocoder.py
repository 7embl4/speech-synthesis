import torch
from torch import nn


class MRF(nn.Module):
    def __init__(self, kernels, dilations, hidden_size=512):
        """
        N res blocks
        each has M layers of leakyrelu and convolution
        """

        super().__init__()
        assert len(kernels) == len(
            dilations
        ), f"Kernels and dilations must be same size ({len(kernels)} != {len(dilations)})"

        self.N, self.M, self.L = len(dilations), len(dilations[0]), len(dilations[0][0])
        self.res_blocks = nn.ModuleList([])
        for n in range(self.N):
            l1 = nn.ModuleList([])
            for m in range(self.M):
                l2 = nn.ModuleList([])
                for l in range(self.L):  # noqa
                    l2.append(
                        nn.Sequential(
                            nn.LeakyReLU(),
                            nn.Conv1d(
                                in_channels=hidden_size,
                                out_channels=hidden_size,
                                kernel_size=kernels[n],
                                dilation=dilations[n][m][l],
                                padding="same",
                            ),
                        )
                    )
                l1.append(l2)
            self.res_blocks.append(l1)

    def forward(self, x: torch.Tensor):
        res = 0.0
        for n in range(self.N):
            _x = x.clone()
            for m in range(self.M):
                res = _x
                for l in range(self.L):  # noqa
                    _x = self.res_blocks[n][m][l](_x)
                _x = res + _x
            res = res + _x
        return res


if __name__ == "__main__":
    mrf = MRF(
        kernels=[3, 7, 11], dilations=[[[1, 1], [3, 1], [5, 1]]] * 3, hidden_size=80
    )
    print(mrf)
    x = torch.rand(1, 80, 4000)
    out = mrf(x)
    print(out.shape)


# class Generator(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         kernels,  # [16, 16, 4, 4]
#         num_layers
#     ):
#         super().__init__()

#         self.layers = nn.ModuleList([])
#         for i in range(num_layers):
#             self.layers.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(
#                         in_channels=in_channels,
#                         out_channels=
#                     )
#                 )
#             )
