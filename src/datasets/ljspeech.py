import os
import random
import tarfile
from pathlib import Path

import pandas as pd
import torchaudio
import torchaudio.functional as F
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

LJSPEECH_DOWNLOAD_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


class LJSpeechDataset(BaseDataset):
    """
    LJSpeech dataset class
    """

    def __init__(
        self,
        data_dir,
        part,
        val_size=150,
        target_sr=22050,
        max_chunk_size=8192,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: local directory either where dataset contains
                or where it going to be downloaded
            part: dataset part (train or val)
            val_size: number of audios for validation
            target_sr: audio sample rate
            max_chunk_size: maximum size of an audio chunk (in number of samples)
        """
        self.data_dir = ROOT_PATH / data_dir
        self.part = part
        self.val_size = val_size
        self.target_sr = target_sr
        self.max_chunk_size = max_chunk_size

        index = self._get_index()
        index = self._split_index(index)
        super().__init__(index, *args, **kwargs)

    def _get_index(self):
        """
        Read or create index
        """
        index_path = self.data_dir / (self.data_dir.name + ".json")
        if index_path.exists():
            index = read_json(index_path)
        else:
            self._load_dataset()
            index = self._create_index()
            write_json(index, index_path)
        return index

    def _split_index(self, index):
        """
        Split index on train and val
        """
        if len(index) < self.val_size:
            raise ValueError(
                f"Validation size ({self.val_size}) is greater than total dataset size ({len(index)})"
            )

        if self.part == "train":
            return index[: -self.val_size]
        elif self.part == "val":
            return index[-self.val_size :]
        else:
            raise ValueError(f"Unsupported part: {self.part}")

    def _load_dataset(self):
        """
        Load dataset from https://keithito.com/LJ-Speech-Dataset/
        """
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            return
        os.makedirs(str(self.data_dir), exist_ok=True)

        print("Downloading dataset...")
        filename = wget.download(LJSPEECH_DOWNLOAD_URL, out=str(self.data_dir.parent))
        print(f"Downloaded to {filename}")

        print("Extracting data...")
        with tarfile.open(filename, "r:bz2") as tar:
            tar.extractall(path=str(self.data_dir.parent))
        os.rename(filename.replace(".tar.bz2", ""), self.data_dir)
        print(f"Data extracted to {self.data_dir}")

        os.remove(filename)

    def _create_index(self):
        """
        Create index if it doesn't exist
        """
        texts_df = pd.read_csv(
            str(self.data_dir / "metadata.csv"),
            sep="|",
            names=["id", "text", "norm_text"],
        )
        wavs_path = self.data_dir / "wavs"

        index = []
        for wav_path in tqdm(wavs_path.iterdir(), desc="Creating index"):
            id = wav_path.stem
            item = texts_df.loc[texts_df["id"] == id]
            text = item["text"].item()
            norm_text = item["norm_text"].item()

            index.append(
                {"filename": str(wav_path), "text": text, "norm_text": norm_text}
            )

        return index

    def __getitem__(self, ind):
        """
        Get item by index
        """
        metadata = self._index[ind]

        audio, sr = torchaudio.load(metadata["filename"])
        audio = audio[:1, :]  # get only first channel

        if sr != self.target_sr:
            audio = F.resample(audio, sr, self.target_sr)

        # get random chunk of audio
        max_chunk_size = (
            self.max_chunk_size if self.part == "train" else self.max_chunk_size * 4
        )
        audio_len = audio.shape[-1]
        if audio_len > max_chunk_size:
            start_ind = random.randint(0, audio_len - max_chunk_size)
            audio = audio[:, start_ind : start_ind + max_chunk_size]

        instance_data = {
            "filename": metadata["filename"],
            "audio": audio,
            "text": metadata["text"],
            "norm_text": metadata["norm_text"],
            "spec": self.get_spectrogram(audio),
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def get_spectrogram(self, audio):
        """
        Special transform to get spectrogram from audio
        """
        if self.instance_transforms is not None:
            spec = self.instance_transforms["get_spectrogram"](audio)
        return spec.squeeze()

    def _assert_index_is_valid(self, index):
        """
        Check if index contains required information
        """
        for entry in index:
            assert "filename" in entry, "Dataset elements must have path to audio"
