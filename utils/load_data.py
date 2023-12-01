from torch.utils.data import DataLoader

from utils.utils import get_blobs_paths_and_names, get_total_number_of_samples
from utils.iterable_blob_dataset import IterableBlobDataset
from utils.split_blob import get_number_of_samples_per_blob


def load_lymphoma_data(batch_size, mode='train', ppb=10000):
    path_to_data = f"/data"
    filename_splits = "splits.csv"
    train_lenghts, val_lenghts, test_lengths = get_number_of_samples_per_blob(path_to_data, filename_splits)
    blobs_paths, blob_names = get_blobs_paths_and_names(path_to_data, number_of_blobs=10)
    total_train_len, total_val_len, total_test_len = get_total_number_of_samples(blob_names, train_lenghts,
                                                                                 val_lenghts, test_lengths)

    print(f"Total number of training samples: {total_train_len}")
    data_iter = IterableBlobDataset(path_to_data, blobs_paths, filename_splits, total_train_len, total_val_len,
                                    total_test_len, mode=mode)
    return DataLoader(data_iter, batch_size=batch_size, drop_last=False, num_workers=5)
