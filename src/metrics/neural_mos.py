import inspect
import os
from pathlib import Path

import torch
import torchaudio.functional as F
from nisqa.NISQA_lib import NISQA, NISQA_DIM
from torch import nn

from src.transforms import MelSpectrogram, MelSpectrogramConfig


class NeuralMOS(nn.Module):
    """NOT WORKING FOR NOW"""

    def __init__(self, weights_path):
        super().__init__()
        self.weights_path = weights_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # mel_cfg = MelSpectrogramConfig(

        # )
        self.mel_spec = MelSpectrogram()
        self.met = self._load_model()
        # self.met = NISQA(cnn_fc_out_h=20, td="lstm", cnn_model="standard", pool="avg")

        self.met.eval()

    @torch.no_grad()
    def forward(self, gen_audio: torch.Tensor, **batch):
        B, C, T = gen_audio.shape
        spec = self.mel_spec(gen_audio)  # [B, C, N_mels, T_spec]
        spec = spec.unsqueeze(1)
        spec = torch.rand(1, 1, 1, 48, 15)
        n_wins = torch.tensor([1] * B)
        res = self.met(spec, n_wins)
        print(res)

    def _load_model(self):
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        model_init_args = list(inspect.signature(NISQA.__init__).parameters.keys())[1:]
        model_args = {}
        for arg, value in checkpoint["args"].items():
            if arg in model_init_args:
                model_args[arg] = value
        print(model_args)
        model = NISQA(**model_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
