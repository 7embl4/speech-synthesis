import os
from pathlib import Path

import torch
import torchaudio
from nisqa.NISQA_model import nisqaModel
from torch import nn


class NeuralMOS(nn.Module):
    """NOT WORKING FOR NOW"""

    def __init__(self, data_dir, batch_size, target_sr):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr

        args = {
            "data_dir": data_dir,
            "pretrained_model": "models/nisqa_tts.tar",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "bs": batch_size,
            "output_dir": data_dir,
        }
        self.metric = nisqaModel(args)

    @torch.no_grad()
    def forward(self, gen_audio: torch.Tensor, **batch):
        os.makedirs(str(self.data_dir), exist_ok=True)

        # temporary save audios
        for audio in gen_audio:
            torchaudio.save(
                str(self.data_dir / "temp_audio_1.wav"), audio, self.target_sr
            )

        # run prediction
        self.metric.predict()

        # load output dict

        # remove temporary stuff

        # return result


if __name__ == "__main__":
    met = NeuralMOS("temp", 8, 22050)
    audio = torch.rand(5, 1, 100000)
    out = met(audio)
    print(out)
