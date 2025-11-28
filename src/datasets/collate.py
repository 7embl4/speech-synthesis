import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    for item in dataset_items:
        for k, v in item.items():
            if k not in result_batch:
                result_batch.setdefault(k, [])

            result_batch[k].append(v)

    # pad tensors
    for k, v in result_batch.items():
        if isinstance(v[0], torch.Tensor):
            result_batch[k] = pad_sequence(v, batch_first=True)

    return result_batch
