import torch
import os
import pandas as pd
import numpy as np


# This file contains all the functions needed to split the data into train, validation and test set.
# The data is split on the slide level, i.e. the patches of one slide are either in the train, validation or test set.
# The split is done by using the information from the 0-th splitted blob.
# The information which slides belong to which set is saved in a csv file which will be used for splitting the other blobs.
# The indices of the patches for each split are returned as a tuple when using the function get_train_val_test_indices().

# Additionaly, this file contains functions to check if there is any overlap between the splits of different blobs or within a blob.
# The functions are called check_overlap_within_splitted_blobs() and check_overlap_between_splitted_blobs() and can be used for example in the main().
# Since the type of test being performed by these functions are time extensive, it is not used by default in split_blob().

# IMPORTANT: Throughout this file, it is assumed that every blob contains patches from the same set of slides.


def get_train_val_test_indices(path_to_blob: str, path_to_splits: str, train_split=0.8, val_split=0.1, test_split=0.1,
                               shuffle=True) -> tuple:
    """
    Splits the data into train, validation and test set.

    Args:
        path_to_blob (str): Path to the blob files, without any file suffix -> needed to get the metadata.
        path_to_splits (str): Path to the csv file containing the information for the splits.
        train_split (float, optional): Percentage of data to use for training. Defaults to 0.8.
        val_split (float, optional): Percentage of data to use for validation. Defaults to 0.1.
        test_split (float, optional): Percentage of data to use for testing. Defaults to 0.1.

    Raises:
        Exception: If the sum of the split percentages does not equal 1.

    Returns:
        tuple: Tuple containing the indices for the train, validation and test set.
    """
    # if sum of split percentages equals 1 -> will be user input later; note: float comparison is not exact
    if np.abs(train_split + val_split + test_split - 1) >= 1e-10:
        raise Exception("Sum of split percentages does not equal 1")
    # split the data into train, validation and test set
    if not os.path.exists(path_to_splits):
        return setup_train_val_test_indices_and_save_splits(path_to_blob, val_split, test_split, path_to_splits,
                                                            shuffle=shuffle)
    else:
        return setup_train_val_test_indices(path_to_blob, path_to_splits, shuffle=shuffle)


def setup_train_val_test_indices_and_save_splits(blob_file: str, val_split: float, test_split: float,
                                                 path_to_splits: str = 'splits.csv', random_seed=None,
                                                 shuffle=True) -> tuple:
    """Splits the data of a blob (normally the 0-th blob) into train, validation and test set, returns them and saves a record of the slides used for the corresponding splits as a csv.

    Args:
        blob_file (str): Path to a blob file -> needed to get the metadata.
        val_split (float): Percentage of slides for the validation set.
        test_split (float): Percentage of slides for the test set.
        path_to_splits (str, optional): Path to the csv file containing the information for the splits. Defaults to 'splits.csv'.
        random_seed (int, optional): Random seed for shuffling the indices. Defaults to None.
        shuffle (bool, optional): If True, the indices will be shuffled. Defaults to True.
    
    Returns:
        tuple: Tuple of the indices for each split in the form (train_indices, val_indices, test_indices).
    """
    # load metadata of one blob and extract from how many different slides the patches are and their corresponding diagnosis
    metadata = pd.read_csv(blob_file + '.csv')
    filenames = metadata['filename'].unique()
    # get diagnosis corresponding to each unique filename above
    diagnosis = [metadata[metadata['filename'] == filename]['diagnosis'].values[0] for filename in filenames]
    if len(filenames) != len(diagnosis):
        raise Exception("Number of filenames does not equal number of diagnosis")
    # zip them together and check if names and diagnosis match when compared to the metadata
    filenames_and_diagnosis = list(zip(filenames, diagnosis))
    for name, label in filenames_and_diagnosis:
        if metadata[metadata['filename'] == name]['diagnosis'].values[0] != label:
            raise Exception("Filename and diagnosis do not match when compared to metadata")

    # get number of slides and sort them to ensure that the splits are always the same -> can shuffle later if wanted
    nr_of_slides = len(filenames_and_diagnosis)
    filenames_and_diagnosis.sort()

    # get filenames for each diagnosis before splitting -> this is needed to ensure that the splits are balanced
    classes = np.unique(diagnosis)
    filenames_per_class = {'%d' % c: [] for c in classes}
    for filename, label in filenames_and_diagnosis:
        filenames_per_class['%d' % label].append(filename)

    # get filenames for each split
    train_filenames, val_filenames, test_filenames = [], [], []
    for c in classes:
        # get filenames for this class
        filenames = filenames_per_class['%d' % c]
        # get filenames for each split for this class
        val_filenames += filenames[: int(np.floor(len(filenames) * val_split))]
        test_filenames += filenames[int(np.floor(len(filenames) * val_split)): int(
            np.floor(len(filenames) * (val_split + test_split)))]
        train_filenames += filenames[
                           int(np.floor(len(filenames) * (val_split + test_split))):]  # rest goes to train set

    # save the splits from the 0-th splitted blob to a csv file which will be used for splitting the other blobs
    save_info_for_splits(filenames_and_diagnosis, train_filenames, val_filenames, test_filenames, path_to_splits)

    # get indices of patches for each split
    train_indices, val_indices, test_indices = get_indices(metadata, train_filenames, val_filenames, test_filenames)

    # convert pandas indices to numpy array and shuffle them if wanted
    train_indices, val_indices, test_indices = convert_to_numpy_and_shuffle(train_indices, val_indices, test_indices,
                                                                            random_seed, shuffle)

    # return the indices
    return train_indices, val_indices, test_indices


def setup_train_val_test_indices(blob_file: str, path_to_splits: str = 'splits.csv', random_seed=None,
                                 shuffle=True) -> tuple:
    """Splits the data of a blob (normally not the 0-th one) into train, validation and test set by using the splits used for the 0-th splitted blob and returns them.

    Args:
        blob_file (str): Path to the blob files -> needed to get the metadata.
        path_to_splits (str, optional): Path to the csv file containing the information for the splits used for the 0-th splitted blob. Defaults to 'splits.csv'.
        random_seed (int, optional): Random seed for shuffling the indices. Defaults to None.
        shuffle (bool, optional): If True, the indices will be shuffled. Defaults to True.
    
    Returns:
        tuple: Tuple of the indices for each split in the form (train_indices, val_indices, test_indices).
    """
    # load metadata of one blob and extract from how many different slides the patches are
    metadata = pd.read_csv(blob_file + '.csv')
    filenames = metadata['filename'].unique()

    # load info for splits
    splits = pd.read_csv(blob_file[0] + path_to_splits)
    splits_filenames = splits['filenames'].unique()
    # check if every slide name from splits is in the blob
    if not np.all([x in filenames for x in splits_filenames]):
        # this shouldn't happen -> if it happens, something went wrong somewhere else probably (e.g sampled from
        # needle biopsy and reached max tries)
        raise Exception("Not every slide name from splits is in the blob")

    # get corresponding filenames for each split
    train_filenames = splits[splits['train'] == 1]['filenames'].values
    val_filenames = splits[splits['validation'] == 1]['filenames'].values
    test_filenames = splits[splits['test'] == 1]['filenames'].values

    # get indices of patches for each split
    train_indices, val_indices, test_indices = get_indices(metadata, train_filenames, val_filenames, test_filenames)

    # convert pandas indices to numpy array and shuffle them if wanted
    train_indices, val_indices, test_indices = convert_to_numpy_and_shuffle(train_indices, val_indices, test_indices,
                                                                            random_seed, shuffle)

    # return the indices
    return train_indices, val_indices, test_indices


##################################################################################################################


def get_number_of_train_val_test_samples(path_to_data: str = '/data', filename_splits: str = 'splits.csv'):
    """Gets the minimal number of patches for each split.

    Args:
        path_to_data (str, optional): Path to the directory where the blobs are stored. Defaults to '/data'.
        filename_splits (str, optional): Name of the csv file containing the information for the splits. Defaults to 'splits.csv'.

    Returns:
        tuple: Tuple containing the minimal number of patches for each split.
    """
    # get names of all blobs within in the data directory without .pt ending
    blob_paths = [path_to_data + "/" + f for f in os.listdir(path_to_data + "/") if f.endswith('.pt')]
    blob_paths = [blob_path[:-3] for blob_path in blob_paths]
    split_paths = [path_to_data + "/" + f[0] + filename_splits for f in os.listdir(path_to_data + "/") if f.endswith('.pt')]
    print(blob_paths)
    # setup data container for the lengths
    total_train_lengths, total_val_lengths, total_test_lengths = 0, 0, 0
    for blob_path, splits_path in zip(blob_paths, split_paths):
        train_indices, val_indices, test_indices = get_train_val_test_indices(blob_path, splits_path)
        # sum up the lengths for each split
        total_train_lengths += len(train_indices)
        total_val_lengths += len(val_indices)
        total_test_lengths += len(test_indices)

    return total_train_lengths, total_val_lengths, total_test_lengths


def get_number_of_samples_per_blob(path_to_data: str = '/data', filename_splits: str = 'splits.csv'):
    """Gets the minimal number of patches for each split.

    Args:
        path_to_data (str, optional): Path to the directory where the blobs are stored. Defaults to '/data'.
        filename_splits (str, optional): Name of the csv file containing the information for the splits. Defaults to 'splits.csv'.

    Returns:
        tuple: Tuple containing the minimal number of patches for each split.
    """
    # get names of all blobs within in the data directory without .pt ending
    blob_paths = [path_to_data + "/" + f for f in os.listdir(path_to_data + "/") if f.endswith('.pt')]
    blob_paths = [blob_path[:-3] for blob_path in blob_paths]
    split_paths = [path_to_data + "/" + f[0] + "splits.csv" for f in os.listdir(path_to_data + "/") if f.endswith('.pt')]
    # setup data container for the lengths
    train_lengths, val_lengths, test_lengths = {}, {}, {}
    for blob_path, split_path in zip(blob_paths, split_paths):
        train_indices, val_indices, test_indices = get_train_val_test_indices(blob_path, split_path)
        # ensure that the blob_name only consists of blob without any prefix
        blob_path = blob_path.split('/')[-1]
        # save number of samples per mode for this blob in dicts 
        train_lengths[blob_path] = len(train_indices)
        val_lengths[blob_path] = len(val_indices)
        test_lengths[blob_path] = len(test_indices)

    return train_lengths, val_lengths, test_lengths


##################################################################################################################
############################### Utilities for train_val_test_split_of_blob() #####################################
##################################################################################################################

def get_nr_of_slides_per_split(nr_of_slides: int, val_split: float, test_split: float) -> tuple:
    """Returns the number of slides for each split.

    Args:
        nr_of_slides (int): Total number of slides present in blob.
        val_split (float): Percentage of slides for the validation set.
        test_split (float): Percentage of slides for the test set.

    Returns:
        tuple: Tuple of number of slides (as ints) for training and validation split (The last one is given trough these two).
    """
    # get number of slides for each split
    nr_of_val_slides = int(np.floor(nr_of_slides * val_split))
    nr_of_test_slides = int(np.floor(nr_of_slides * test_split))
    nr_of_train_slides = nr_of_slides - nr_of_val_slides - nr_of_test_slides  # rest goes to train set

    # check if sum of splits equals number of slides
    if nr_of_train_slides + nr_of_val_slides + nr_of_test_slides != nr_of_slides:
        raise Exception("Sum of splits does not equal number of slides")
    return nr_of_train_slides, nr_of_val_slides


def save_info_for_splits(filenames_and_diagnosis: list, train_filenames: list, val_filenames: list,
                         test_filenames: list, path_to_splits: str = 'splits.csv') -> None:
    """Saves the information for the splits done in the 0-th splitted blob as a csv file.
    
    Args:
        filenames_and_diagnosis (list): List of all filenames and diagnosis in the blob [(name1,d1), (name2,d2), ...].
        train_filenames (list): List of filenames for the training set.
        val_filenames (list): List of filenames for the validation set.
        test_filenames (list): List of filenames for the test set.
        path_to_splits (str, optional): Path to the csv file containing the information for the splits. Defaults to 'splits.csv'.
    """
    # unpack the filenames and diagnosis
    filenames, diagnosis = zip(*filenames_and_diagnosis)
    # create a dataframe with the filenames and the corresponding split (one-hot encoded)
    nr_of_slides = len(filenames)
    splits = pd.DataFrame({'filenames': filenames, 'diagnosis': diagnosis,
                           'train': np.zeros(nr_of_slides), 'validation': np.zeros(nr_of_slides),
                           'test': np.zeros(nr_of_slides)})
    # the .loc method locats the rows that meet the condition specified in the square brackets (splits[
    # 'filenames'].isin(train_filenames)), and then it sets the 'train' column for those rows to 1.
    splits.loc[splits['filenames'].isin(train_filenames), 'train'] = 1
    splits.loc[splits['filenames'].isin(val_filenames), 'validation'] = 1
    splits.loc[splits['filenames'].isin(test_filenames), 'test'] = 1

    # check that there is one 1 per row (Note: axis=1 means that we sum over the columns)
    if not np.all(splits[['train', 'validation', 'test']].sum(axis=1) == 1):
        raise Exception(
            "There is at least one row where the sum of splits is not 1, i.e. one slide is in more than one split")

    # make sure that ints are saved as ints and not floats
    splits['train'] = splits['train'].astype(int)
    splits['validation'] = splits['validation'].astype(int)
    splits['test'] = splits['test'].astype(int)
    # save the splits to a csv file
    splits.to_csv(path_to_splits, index=False)


def get_indices(metadata: pd.DataFrame, train_filenames: list, val_filenames: list, test_filenames: list) -> tuple:
    """Returns the indices of patches for each split.

    Args:
        metadata (pd.DataFrame): Metadata of the patches.
        train_filenames (list): List of filenames for the training set.
        val_filenames (list): List of filenames for the validation set.
        test_filenames (list): List of filenames for the test set.

    Returns:
        tuple: Tuple of indices for each split.
    """
    # get indices of patches for each split
    train_indices = metadata[metadata['filename'].isin(
        train_filenames)].index  # .index returns the indices of the rows that meet the condition specified in the
    # square brackets
    val_indices = metadata[metadata['filename'].isin(val_filenames)].index
    test_indices = metadata[metadata['filename'].isin(test_filenames)].index

    # check if sum of indices equals number of patches
    if len(train_indices) + len(val_indices) + len(test_indices) != len(metadata):
        print(len(train_indices), len(val_indices), len(test_indices), len(metadata))
        raise Exception("Sum of train, validation and test indices does not equal total number of patches")

    # check if the indices are within the range of the metadata
    if np.any(train_indices >= len(metadata)) or np.any(val_indices >= len(metadata)) or np.any(
            test_indices >= len(metadata)):
        raise Exception("At least one index is out of range of the metadata")

    # check if the indices are unique (set removes duplicates)
    if len(set(train_indices)) != len(train_indices) or len(set(val_indices)) != len(val_indices) or len(
            set(test_indices)) != len(test_indices):
        raise Exception("At least one index list contains duplicates")

    # check if one index is in more than one split -> avoid data leakage!
    train_n_val = set(train_indices).intersection(set(val_indices))
    train_n_test = set(train_indices).intersection(set(test_indices))
    val_n_test = set(val_indices).intersection(set(test_indices))
    if len(train_n_val) != 0 or len(train_n_test) != 0 or len(val_n_test) != 0:
        raise Exception("At least one index is in more than one split")
    return train_indices, val_indices, test_indices


def split_data(data: torch.Tensor, train_indices: list, val_indices: list, test_indices: list,
               pedantic: bool = False) -> tuple:
    """Splits the data loaded according to the indices.

    Args:
        data (torch.Tensor): By custom dataloader loaded data to split.
        train_indices (list): List of indices for the training set.
        val_indices (list): List of indices for the validation set.
        test_indices (list): List of indices for the test set.
        pedantic (bool, optional): If True, the function will check if any sample from one set is in another set to avoid data leakage. This will be time consuming. Defaults to False.

    Returns:
        tuple: Tuple of splitted data (tuple of tuples).
    """
    # unpack the data
    patches, labels = data
    # use the indices to split the data
    train_data = (patches[train_indices], labels[train_indices])
    val_data = (patches[val_indices], labels[val_indices])
    test_data = (patches[test_indices], labels[test_indices])

    # check if the shapes match within the tuples
    if train_data[0].shape[0] != train_data[1].shape[0] or val_data[0].shape[0] != val_data[1].shape[0] or \
            test_data[0].shape[0] != test_data[1].shape[0]:
        raise Exception("Number of patches does not match the number of labels within the resulting splitted data")
    # check if the shapes match with the number of indices
    if train_data[0].shape[0] != len(train_indices) or val_data[0].shape[0] != len(val_indices) or test_data[0].shape[
        0] != len(test_indices):
        raise Exception("Number of indices does not match the shape of the resulting splitted data")
    # check if any sample from one set is in another set -> avoid data leakage. 
    if pedantic == True:
        check_overlap_within_one_splitted_blob(train_data[0], val_data[0], test_data[0])
    return train_data, val_data, test_data


def convert_to_numpy_and_shuffle(train_indices: pd.Index, val_indices: pd.Index, test_indices: pd.Index,
                                 random_seed=None, shuffle=True) -> tuple:
    """Converts the indices to numpy arrays and shuffles them if wanted.

    Args:
        train_indices (pd.Index): Indices of the training set.
        val_indices (pd.Index): Indices of the validation set.
        test_indices (pd.Index): Indices of the test set.
        random_seed (int, optional): Random seed for shuffling the indices. Defaults to None.
        shuffle (bool, optional): If True, the indices will be shuffled. Defaults to True.

    Returns:
        tuple: Tuple of the indices as numpy arrays.
    """
    # convert to numpy array
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    # shuffle the indices if wanted with given random seed
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
    return train_indices, val_indices, test_indices


##################################################################################################################
####################################### Check for any kind of data leakage #######################################
##################################################################################################################

def check_for_overlap_within_splitted_blobs(path_to_data: str, path_to_splits: str) -> None:
    """Tests if samples from the same blob and different splits have overlap.
    
    Args:
        path_to_data (str): Path to the directory where the blobs are stored.
        path_to_splits (str): Path to the csv file containing the information for the splits.
    """
    # count the .pt files located in path_to_data
    nr_of_blobs = sum([f.endswith('.pt') for f in os.listdir(path_to_data)])

    # iterate over all blobs and test if they have overlap
    for k in range(nr_of_blobs):
        print(f"Testing overlap within BLOB {k}")
        # get the indices for each split for this blob
        blob_name = f'{k}blob'
        blob_file = os.path.join(path_to_data, blob_name)
        train_indices, val_indices, test_indices = get_train_val_test_indices(blob_file, path_to_splits)

        # load a blob and split it into train, validation and test set
        data = torch.load(blob_file + '.pt')
        train_data, val_data, test_data = split_data(data, train_indices, val_indices, test_indices)

        # check if samples from the same blob and different splits have overlap -> give only the patches as input
        check_overlap_within_one_splitted_blob(train_data[0], val_data[0], test_data[0])

    print("No overlap within the splits of the same blob found!")


##################################################################################################################
######################## helper function for test_overlap_within_splitted_blobs() ################################
##################################################################################################################

def check_overlap_within_one_splitted_blob(train_patches, val_patches, test_patches) -> None:
    """Checks if samples from the same blob and different splits have overlap and raises an exception if so."""
    # Note: .tolist() is needed to convert the flattened tensor to a list, otherwise the set() function will yield a set of tensors and tuple is needed to make the elements hashable
    train_n_val = set([tuple(patch.flatten().tolist()) for patch in train_patches]).intersection(
        set([tuple(patch.flatten().tolist()) for patch in val_patches]))
    train_n_test = set([tuple(patch.flatten().tolist()) for patch in train_patches]).intersection(
        set([tuple(patch.flatten().tolist()) for patch in test_patches]))
    val_n_test = set([tuple(patch.flatten().tolist()) for patch in val_patches]).intersection(
        set([tuple(patch.flatten().tolist()) for patch in test_patches]))

    if len(train_n_val) != 0 or len(train_n_test) != 0 or len(val_n_test) != 0:
        raise Exception("At least one sample is in more than one split")


##################################################################################################################

def check_overlap_between_splitted_blobs(path_to_data: str, path_to_splits: str) -> None:
    """Tests if samples from different blobs and different splits have overlap.
    
    Args:
        path_to_data (str): Path to the directory where the blobs are stored.
        path_to_splits (str): Path to the csv file containing the information for the splits.
    """
    # count the .pt files located in path_to_data
    nr_of_blobs = sum([f.endswith('.pt') for f in os.listdir(path_to_data)])

    # iterate over all combinations of blobs and test if they have overlap
    for k in range(nr_of_blobs):
        # get the indices for each split for this blob
        blob0_name = f'{k}blob'
        blob0_file = os.path.join(path_to_data, blob0_name)
        train_indices, val_indices, test_indices = get_train_val_test_indices(blob0_file, path_to_splits)

        # load a blob and split it into train, validation and test set
        data = torch.load(blob0_file + '.pt')
        train_data_0, val_data_0, test_data_0 = split_data(data, train_indices, val_indices, test_indices)

        for i in range(k + 1, nr_of_blobs):
            print(f"Testing overlap between different splits of BLOB {k} and BLOB {i}")
            # get the indices for each split for this blob
            blob1_name = f'{i}blob'
            blob1_file = os.path.join(path_to_data, blob1_name)
            train_indices_1, val_indices_1, test_indices_1 = get_train_val_test_indices(blob1_file, path_to_splits)

            # load a blob and split it into train, validation and test set
            data = torch.load(blob1_file + '.pt')
            train_data_1, val_data_1, test_data_1 = split_data(data, train_indices_1, val_indices_1, test_indices_1)

            # check if samples from different blobs and different splits have overlap -> give only the patches as input
            check_overlap_between_two_splitted_blobs(train_data_0[0], val_data_0[0], test_data_0[0], train_data_1[0],
                                                     val_data_1[0], test_data_1[0])

    print("No overlap between different splits of different blobs found!")


##################################################################################################################
######################## helper function for test_overlap_between_splitted_blobs() ###############################
##################################################################################################################

def check_overlap_between_two_splitted_blobs(train_patches_0, val_patches_0, test_patches_0, train_patches_1,
                                             val_patches_1, test_patches_1) -> None:
    """Checks if samples from two different blobs and different splits have overlap and raises an exception if so."""
    # Note: .tolist() is needed to convert the flattened tensor to a list, otherwise the set() function will yield a set of tensors and tuple is needed to make the elements hashable
    train_0_n_val_1 = set([tuple(patch.flatten().tolist()) for patch in train_patches_0]).intersection(
        set([tuple(patch.flatten().tolist()) for patch in val_patches_1]))
    train_0_n_test_1 = set([tuple(patch.flatten().tolist()) for patch in train_patches_0]).intersection(
        set([tuple(patch.flatten().tolist()) for patch in test_patches_1]))
    val_0_n_test_1 = set([tuple(patch.flatten().tolist()) for patch in val_patches_0]).intersection(
        set([tuple(patch.flatten().tolist()) for patch in test_patches_1]))
    val_0_n_train_1 = set([tuple(patch.flatten().tolist()) for patch in val_patches_0]).intersection(
        set([tuple(patch.flatten().tolist()) for patch in train_patches_1]))
    test_0_n_train_1 = set([tuple(patch.flatten().tolist()) for patch in test_patches_0]).intersection(
        set([tuple(patch.flatten().tolist()) for patch in train_patches_1]))
    test_0_n_val_1 = set([tuple(patch.flatten().tolist()) for patch in test_patches_0]).intersection(
        set([tuple(patch.flatten().tolist()) for patch in val_patches_1]))

    if len(train_0_n_val_1) != 0 or len(train_0_n_test_1) != 0 or len(val_0_n_test_1) != 0 or len(
            val_0_n_train_1) != 0 or len(test_0_n_train_1) != 0 or len(test_0_n_val_1) != 0:
        raise Exception("At least one sample is in more than one split")


##################################################################################################################


def get_number_of_samples_per_blob_for_testing(path_to_training_output: str, path_to_data: str = '/data',
                                               filename_splits: str = 'splits.csv'):
    """Gets the minimal number of patches for each split for testing. Note: During testing we always take the splits file saved while training in the output directory of the corresponding training run.

    Args:
        path_to_training_output (str): Path to the directory where the training output is stored. The splits.csv file should be located there.
        path_to_data (str, optional): Path to the directory where the blobs are stored. Defaults to '/data'.
        filename_splits (str, optional): Name of the csv file containing the information for the splits. Defaults to 'splits.csv'.

    Returns:
        tuple: Tuple containing the minimal number of patches for each split.
    """
    # get names of all blobs within in the data directory without .pt ending
    blob_paths = [path_to_data + "/" + f for f in os.listdir(path_to_data + "/") if f.endswith('.pt')]
    blob_paths = [blob_path[:-3] for blob_path in blob_paths]
    path_to_splits = path_to_data + "/" + filename_splits
    # setup data container for the lengths
    train_lengths, val_lengths, test_lengths = {}, {}, {}
    for blob_name in blob_paths:
        train_indices, val_indices, test_indices = get_train_val_test_indices_for_testing(blob_name, path_to_splits)
        # ensure that the blob_name only consists of blob without any prefix
        blob_name = blob_name.split('/')[-1]
        # save number of samples per mode for this blob in dicts
        train_lengths[blob_name] = len(train_indices)
        val_lengths[blob_name] = len(val_indices)
        test_lengths[blob_name] = len(test_indices)

    return train_lengths, val_lengths, test_lengths


def get_train_val_test_indices_for_testing(path_to_blob: str, path_to_splits: str, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True) -> tuple:
    """
    Splits the data into train, validation and test set for testing. Note: During testing we always take the splits file saved while training in the output directory of the corresponding training run.

    Args:
        path_to_blob (str): Path to the blob files, without any file suffix -> needed to get the metadata.
        path_to_splits (str): Path to the csv file containing the information for the splits.
        train_split (float, optional): Percentage of data to use for training. Defaults to 0.8.
        val_split (float, optional): Percentage of data to use for validation. Defaults to 0.1.
        test_split (float, optional): Percentage of data to use for testing. Defaults to 0.1.

    Raises:
        Exception: If the sum of the split percentages does not equal 1.

    Returns:
        tuple: Tuple containing the indices for the train, validation and test set.
    """
    # if sum of split percentages equals 1 -> will be user input later; note: float comparison is not exact
    if np.abs(train_split + val_split + test_split - 1) >= 1e-10:
        raise Exception("Sum of split percentages does not equal 1")
    # split the data into train, validation and test set
    if not os.path.exists(path_to_splits):
        raise Exception("No splits.csv file found. It should be saved during training in the output directory of the corresponding training run.")
    else:
        return setup_train_val_test_indices(path_to_blob, path_to_splits, shuffle=shuffle)

"""
docker run --shm-size=100gb --gpus all -it -u `id -u $USER` --rm -v /sybig/home/fto/code/lymphoma/DDP:/mnt -v /sybig/home/fto/preprocessed_data/sandbox_data/10_blobs_1000_ppb:/data fto_cnn python3 /mnt/split_blob.py 
"""


def main():
    # # load a blob and split it into train, validation and test set
    # path_to_data = '/data'
    path_to_data = "/Users/ferdinandtolkes/data/sandbox_data/10_blobs_1000_ppb"
    path_to_splits = path_to_data + '/splits.csv'

    # functions below are used for checking if the splitting works as intended
    check_overlap_between_splitted_blobs(path_to_data, path_to_splits)
    check_for_overlap_within_splitted_blobs(path_to_data, path_to_splits)


if __name__ == '__main__':
    # main()
    pass
