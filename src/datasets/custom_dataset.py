import os

from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


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
        if self.data_dir.exists():
            return

        os.makedirs(str(self.data_dir))
        # loading using yadisk tool

    def _create_index(self):
        texts_path = self.data_dir / "transcriptions"

        index = []
        for text_path in tqdm(texts_path.iterdir()):
            with text_path.open() as f:
                transcript = f.readline()

            index.append({"filename": str(text_path), "text": transcript})

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
