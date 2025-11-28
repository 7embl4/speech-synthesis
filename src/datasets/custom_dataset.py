import os
import zipfile

from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, download_from_yadisk, read_json, write_json


class CustomUtteranceDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        self.data_dir = ROOT_PATH / data_dir

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
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            return

        os.makedirs(str(self.data_dir), exist_ok=True)
        with (ROOT_PATH / ".env").open() as env_file:
            env_vars = {
                line.split("=")[0]: line.split("=")[1] for line in env_file.readlines()
            }
        data_url = env_vars["CUSTOM_DATASET_URL"]
        save_path = str(self.data_dir / (self.data_dir.name + ".zip"))
        download_from_yadisk(data_url, save_path)

        print("Unziping data...")
        with zipfile.ZipFile(save_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

    def _create_index(self):
        index = []
        print("Index not found, creating...")
        for parent, dirs, files in self.data_dir.walk():
            for file in files:
                if file.endswith(".txt"):
                    with (parent / file).open() as f:
                        text = f.readline()

                    index.append({"filename": str(parent / file), "text": text})
        print("Index created")

        return index

    def __getitem__(self, ind):
        metadata = self._index[ind]

        instance_data = {
            "filename": metadata["filename"],
            "text": metadata["text"],
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def _assert_index_is_valid(self, index):
        for entry in index:
            assert (
                "filename" in entry
            ), "Dataset elements must have path to transcriptions"
            assert "text" in entry, "Dataset elements must have transcriptions"
