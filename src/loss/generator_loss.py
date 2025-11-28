import torch
import torch.nn.functional as F
from torch import nn

from src.transforms import MelSpectrogram


class GeneratorLoss(nn.Module):
    def __init__(self, lambda_fm=2, lambda_mel=45):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.mel_spec = MelSpectrogram()

    def forward(
        self,
        spec,
        gen_audio,
        mpd_gs,
        msd_gs,
        mpd_r_fmaps,
        mpd_g_fmaps,
        msd_r_fmaps,
        msd_g_fmaps,
        **batch
    ):
        adv_loss = self._calc_adv_loss(mpd_gs, msd_gs)
        fm_loss = self._calc_fm_loss(mpd_r_fmaps, mpd_g_fmaps, msd_r_fmaps, msd_g_fmaps)
        mel_loss = self._calc_mel_loss(spec, gen_audio)

        return adv_loss + self.lambda_fm * fm_loss + self.lambda_mel * mel_loss

    def _calc_adv_loss(self, mpd_gs, msd_gs):
        mpd_loss = 0.0
        for i in range(len(mpd_gs)):
            mpd_loss = mpd_loss + torch.mean((mpd_gs[i] - 1) ** 2)

        msd_loss = 0.0
        for i in range(len(msd_gs)):
            msd_loss = msd_loss + torch.mean((msd_gs[i] - 1) ** 2)

        return mpd_loss + msd_loss

    def _calc_fm_loss(self, mpd_r_fmaps, mpd_g_fmaps, msd_r_fmaps, msd_g_fmaps):
        mpd_loss = 0.0
        N, M = len(mpd_r_fmaps), len(mpd_r_fmaps[0])
        for i in range(N):
            for j in range(M):
                mpd_loss = mpd_loss + torch.mean(
                    torch.abs(mpd_r_fmaps[i][j] - mpd_g_fmaps[i][j])
                )

        msd_loss = 0.0
        N, M = len(msd_r_fmaps), len(msd_r_fmaps[0])
        for i in range(N):
            for j in range(M):
                msd_loss = msd_loss + torch.mean(
                    torch.abs(msd_r_fmaps[i][j] - msd_g_fmaps[i][j])
                )

        return mpd_loss + msd_loss

    def _calc_mel_loss(self, real_spec, gen_audio):
        gen_spec = self.mel_spec(gen_audio)
        real_spec, gen_spec = self._pad_to_equal(real_spec, gen_spec)

        return torch.mean(torch.abs(real_spec - gen_spec))

    def _pad_to_equal(self, real_spec: torch.Tensor, gen_spec: torch.Tensor):
        diff = abs(real_spec.shape[-1] - gen_spec.shape[-1])

        if real_spec.shape[-1] < gen_spec.shape[-1]:
            real_spec = F.pad(real_spec, (0, diff), mode="reflect")
        else:
            real_spec = F.pad(gen_spec, (0, diff), mode="reflect")

        return real_spec, gen_spec
