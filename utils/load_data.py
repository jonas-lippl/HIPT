import os
import random

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from utils.utils import get_blobs_paths_and_names, get_total_number_of_samples
from utils.iterable_blob_dataset import IterableBlobDataset
from utils.split_blob import get_number_of_samples_per_blob


class CustomFolderDs(Dataset):
    def __init__(self, path, names, transform=None):
        self.names = names
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        data = torch.load(self.path + "/" + self.names[idx])
        if self.transform:
            return self.transform(data[0].div(255.0)), data[1]
        return data[0].div(255.0), data[1]


class EmbeddingDS(Dataset):
    def __init__(self, path, names):
        self.names = names
        self.path = path

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        data = torch.load(self.path + "/" + self.names[idx])
        # Being able to handel int labels and torch tensor labels
        if type(data[1]) is int:
            return data[0], torch.tensor(data[1])  #, self.names[idx]
        # Return name of slide for debugging purposes
        return data[0], data[1]  # , self.names[idx]


def load_lymphoma_data(batch_size, mode='train'):
    path_to_data = f"/data"
    filename_splits = "splits.csv"
    train_lenghts, val_lenghts, test_lengths = get_number_of_samples_per_blob(path_to_data, filename_splits)
    number_of_blobs = len([file for file in os.listdir(path_to_data) if file.endswith(".pt")])
    print(f"Number of blobs: {number_of_blobs}")
    blobs_paths, blob_names = get_blobs_paths_and_names(path_to_data, number_of_blobs=number_of_blobs)
    total_train_len, total_val_len, total_test_len = get_total_number_of_samples(blob_names, train_lenghts,
                                                                                 val_lenghts, test_lengths)

    print(f"Total number of training samples: {total_train_len}")
    data_iter = IterableBlobDataset(path_to_data, blobs_paths, filename_splits, total_train_len, total_val_len,
                                    total_test_len, mode=mode)
    return DataLoader(data_iter, batch_size=batch_size, drop_last=False, num_workers=2, timeout=600, prefetch_factor=2)


# NOTE: This took 162.23 seconds per epoch for 3000 patches
def load_lymphoma_data_single_patches(batch_size, mode='train'):
    path_to_data = f"/data"
    patches = [file for file in os.listdir(path_to_data)]
    train_patches = patches[:int(0.8 * len(patches))]
    val_patches = patches[int(0.8 * len(patches)):]
    print(f"Total number of training samples: {len(train_patches)}")
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CustomFolderDs(path_to_data, train_patches,
                             transform=transform) if mode == 'train' else CustomFolderDs(path_to_data,
                                                                                         val_patches,
                                                                                         transform=transform)
    return DataLoader(dataset, sampler=DistributedSampler(dataset), batch_size=batch_size, drop_last=False,
                      num_workers=4)


def load_lymphoma_data_single_patch_embeddings(batch_size, mode='train'):
    if mode == 'train':
        path_to_data = f"/data/single_4096px_2048mu_train"
    else:
        path_to_data = f"/data/single_4096px_2048mu_test"
    patches = [file for file in os.listdir(path_to_data)]
    print(f"Total number of samples: {len(patches)}")
    random.shuffle(patches)
    # train_patches = patches[:int(0.8 * len(patches))]
    # val_patches = patches[int(0.8 * len(patches)):]
    # if mode == 'train':
    #     print(f"Total number of training samples: {len(train_patches)}")
    # else:
    #     print(f"Total number of test samples: {len(val_patches)}")
    dataset = EmbeddingDS(path_to_data, patches)
    # dataset = EmbeddingDS(path_to_data, train_patches) if mode == 'train' else EmbeddingDS(path_to_data, val_patches)
    return DataLoader(dataset, sampler=DistributedSampler(dataset), batch_size=batch_size, drop_last=False,
                      num_workers=4)


def load_lymphoma_data_WSI_embeddings():
    path_to_data = f"/data/WSI_patches_4096px_2048mu_4k_embeddings"
    with open("/data/test_slides.txt", "r") as f:
        test_slides = f.readlines()
    test_slides = [slide.replace("\n", "") for slide in test_slides]
    wsi_patches = [file for file in os.listdir(path_to_data) if file.split(".")[0] in test_slides]
    print(f"Total number of samples: {len(wsi_patches)}")
    dataset = EmbeddingDS(path_to_data, wsi_patches)
    return DataLoader(dataset, batch_size=1, drop_last=False, num_workers=4)
