import torch
import torch.nn.functional as F


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
    for key, list in result_batch.items():
        if isinstance(list[0], torch.Tensor):
            max_len = max([l.shape[-1] for l in list])  # noqa
            new_list = []
            for tensor in list:
                diff = max_len - tensor.shape[-1]
                new_list.append(F.pad(tensor, pad=(0, diff), value=0))

            result_batch[key] = torch.stack(new_list)

    return result_batch
