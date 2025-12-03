import os
import shutil

import torch
import torchaudio

from src.metrics.base_metric import BaseMetric
from src.metrics.nisqa.NISQA_lib import predict_mos
from src.metrics.nisqa.NISQA_model import nisqaModel
from src.utils.io_utils import ROOT_PATH


class NeuralMOS(BaseMetric):
    """
    Calcs MOS using NISQA model
    """

    def __init__(
        self, model_path, mode="predict_dir", target_sr=22050, *agrs, **kwargs
    ):
        super().__init__(*agrs, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_dir = ROOT_PATH / "data" / "temp"

        self.model_path = model_path
        self.target_sr = target_sr
        self.mode = mode

    @torch.no_grad()
    def __call__(self, gen_audio: torch.Tensor, **batch):
        """
        Saves generated audio and calculates metric
        """
        B, C, T = gen_audio.shape

        # save generated audio
        os.makedirs(str(self.temp_dir), exist_ok=True)
        for i in range(B):
            torchaudio.save(
                str(self.temp_dir / f"gen_audio_{i}.wav"),
                gen_audio[i].detach().cpu(),
                self.target_sr,
            )

        # load model
        nisqa_model, args = self._load_model(self.model_path)

        # make predictions
        predict_mos(
            nisqa_model.model,
            nisqa_model.ds_val,
            args["tr_bs_val"],
            self.device,
            num_workers=args["tr_num_workers"],
        )
        res_df = nisqa_model.ds_val.df
        total_mos = res_df["mos_pred"].mean()

        shutil.rmtree(self.temp_dir)
        return total_mos

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)

        args = {}
        args.update(checkpoint["args"])
        args.update(
            {
                "mode": self.mode,
                "pretrained_model": model_path,
                "data_dir": self.temp_dir,
                "output_dir": None,
                "tr_bs_val": 1,
                "tr_num_workers": 0,
                "tr_device": self.device,
                "ms_channel": None,
                "ms_sr": self.target_sr,
                "ms_fmax": self.target_sr // 2,
            }
        )

        return nisqaModel(args), args
