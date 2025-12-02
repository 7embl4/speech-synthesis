import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mpd_rs, mpd_gs, msd_rs, msd_gs, **batch):
        mpd_loss = 0.0
        for i in range(len(mpd_rs)):
            mpd_loss = mpd_loss + torch.mean((mpd_rs[i] - 1) ** 2 + mpd_gs[i] ** 2)

        msd_loss = 0.0
        for i in range(len(msd_rs)):
            msd_loss = msd_loss + torch.mean((msd_rs[i] - 1) ** 2 + msd_gs[i] ** 2)

        total_loss = mpd_loss + msd_loss
        return {"d_loss": total_loss}
