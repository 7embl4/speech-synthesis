import os

import torch
import torchaudio
import utmosv2
from torch import nn

from src.utils.io_utils import ROOT_PATH


class NeuralMOS(nn.Module):
    """
    Calcs MOS using UTMOSv2 model
    """

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = utmosv2.create_model(pretrained=True, device=self.device)
        self.temp_dir = ROOT_PATH / "data/temp"

    @torch.no_grad()
    def forward(self, gen_audio: torch.Tensor, **batch):
        """
        Saves generated audio and calculates metric
        """
        B, C, T = gen_audio.shape
        os.makedirs(str(self.temp_dir), exist_ok=True)
        for i in range(B):
            torchaudio.save(f"data/temp/gen_audio_{i}.wav", gen_audio[i], 22050)

        out = self.model.predict(input_dir=str(self.temp_dir), verbose=False)
        res = 0
        for r in out:
            print(r["predicted_mos"])
            res += r["predicted_mos"]

        os.remove(self.temp_dir)
        res = torch.tensor(res)
        return res.mean()
