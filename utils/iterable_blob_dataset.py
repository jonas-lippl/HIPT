import io
import math
import time

import torch

from torch.utils.data import DistributedSampler
from torch.utils.data import IterableDataset, TensorDataset
from itertools import chain
from utils.split_blob import get_train_val_test_indices


class IterableBlobDataset(IterableDataset):
    def __init__(
            self,
            data_dir: str,
            file_paths: list,
            splits_filename: str,
            total_train_len: int,
            total_val_len: int,
            total_test_len: int,
            mode: str = "train",
            transform=None,
            target_transform=None
    ) -> None:
        self.data_dir = data_dir
        self.file_paths = file_paths
        self.splits_filename = splits_filename
        self.total_train_len = total_train_len
        self.total_val_len = total_val_len
        self.total_test_len = total_test_len
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.epoch = 0

    def __len__(self):
        """ Return the length of min_train_len, min_val_len or min_test_len depending on the current mode. """
        if self.mode == "train":
            return self.total_train_len
        elif self.mode == "val":
            return self.total_val_len
        elif self.mode == "test":
            return self.total_test_len
        else:
            raise ValueError("Mode must be one of: train, val, test")

    # this processes one blob at a time
    def process_data(self, data):
        patches, labels = torch.load(data)
        # convert patches to float32 in [0,1] -> otherwise error in transforms.Normalize
        patches = patches.div(255.0)
        if self.transform:
            patches = self.transform(patches)
        if self.target_transform:
            labels = self.target_transform(labels)

        # Prepare indices for train test val split
        blob_index = data.split('blob')[0][-1]
        path_to_splits = self.data_dir + '/' + blob_index + self.splits_filename
        train_indices, val_indices, test_indices = get_train_val_test_indices(data[:-3], path_to_splits,
                                                                              train_split=0.8, val_split=0.1,
                                                                              test_split=0.1,
                                                                              shuffle=False)  # remove .pt ending
        if self.mode == "train":
            # take only the first min_train_len indices -> indices were already shuffled
            indices = train_indices
        elif self.mode == "val":
            indices = val_indices
        elif self.mode == "test":
            indices = test_indices
        else:
            raise ValueError("Mode must be one of: train, val, test")

        # sample using the indices for the current mode
        patches = patches[indices]
        labels = labels[indices]

        # create dataset
        dataset = TensorDataset(patches, labels)

        # create sampler for distributed training
        sampler = DistributedSampler(dataset, shuffle=True)
        sampler.set_epoch(self.epoch)  # needed when using ddp
        for sample in sampler:
            yield dataset[sample]

    def set_epoch(self, epoch):
        """Sets the epoch needed for the sampler. Used to access epoch from the trainer class."""
        self.epoch = epoch

    def get_stream(self, file_paths):
        return chain.from_iterable(map(self.process_data, chain(
            file_paths)))  # TODO: can we use cycle instead of chain to not exit the loader for multiple epochs -> maybe using islice() (see https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.get_stream(self.file_paths)
        else:
            per_worker = int(
                math.ceil(len(self.file_paths) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths))
            print(f"Worker {worker_id} | Iter start: {iter_start} | Iter end: {iter_end}")
            return self.get_stream(self.file_paths[iter_start:iter_end])
