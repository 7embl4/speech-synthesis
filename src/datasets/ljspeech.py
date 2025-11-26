import os
import tarfile

import pandas as pd
import torchaudio
import torchaudio.functional as F
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LJSpeechDataset(BaseDataset):
    def __init__(self, data_dir, target_sr=22050, *args, **kwargs):
        self.data_dir = ROOT_PATH / data_dir
        self.target_sr = target_sr

        index = self._get_index()
        super().__init__(index, *args, **kwargs)

    def _get_index(self):
        index_path = self.data_dir / (self.data_dir.name + ".json")
        if index_path.exists():
            index = read_json(index_path)
        else:
            self._load_dataset()
            index = self._create_index()
            write_json(index, index_path)
        return index

    def _load_dataset(self):
        # TODO: make it work
        if self.data_dir.exists():
            return

        os.makedirs(str(self.data_dir))
        with (ROOT_PATH / ".env").open() as env_file:
            env_vars = {
                line.split("=")[0]: line.split("=")[1] for line in env_file.readlines()
            }

        if "LJSPEECH_DATASET_URL" in env_vars:
            download_url = env_vars["LJSPEECH_DATASET_URL"]
        else:
            raise ValueError(
                "Provide link in .env file with name: 'LJSPEECH_DATASET_URL'"
            )

        filename = wget.download(download_url, out=str(self.data_dir.parent))
        print(f"Downloaded to {str(self.data_dir / filename)}")

        print("Extracting data...")
        with tarfile.open(filename, "r:bz2") as tar:
            tar.extractall(path=str(self.data_dir))

    def _create_index(self):
        texts_df = pd.read_csv(
            str(self.data_dir / "metadata.csv"),
            sep="|",
            names=["id", "text", "norm_text"],
        )
        wavs_path = self.data_dir / "wavs"

        index = []
        for wav_path in tqdm(wavs_path.iterdir()):
            id = wav_path.stem
            transcript = texts_df.loc[texts_df["id"] == id]["norm_text"].item()

            index.append({"filename": str(wav_path), "text": transcript})

        return index

    def __getitem__(self, ind):
        metadata = self._index[ind]

        audio, sr = torchaudio.load(metadata["filename"])
        audio = audio[:1]  # get only first channel

        if sr != self.target_sr:
            audio = F.resample(audio, sr, self.target_sr)

        instance_data = {
            "filename": metadata["filename"],
            "audio": audio,
            "text": metadata["text"],
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def _assert_index_is_valid(self, index):
        for entry in index:
            assert "filename" in entry, "Dataset elements must have path to audio"
