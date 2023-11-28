import os

import numpy as np

LABEL_MAP = {
    0: "CLL",  # Chronic Lymphocytic Leukemia
    1: "DLBCL",  # Diffuse Large B-Cell Lymphoma
    2: "FL",  # Follicular Lymphoma
    3: "HL",  # Hodgkin Lymphoma
    4: "LYM",  # Lymphadenitis
    5: "MCL",  # Mantle Cell Lymphoma
}


def get_blobs_paths_and_names(path_to_data: str, number_of_blobs: int) -> tuple:
    """
    Retrieves the paths and names of a specifed number of blobs in a given directory.

    Args:
        path_to_data (str): Path to the directory where the blobs are stored.
        number_of_blobs (int): Number of blobs to be used for training.

    Returns:
        tuple: Tuple containing the paths and names of the blobs.
    """
    # get all paths to blobs
    blobs_paths = [path_to_data + "/" + f for f in os.listdir(path_to_data + "/") if f.endswith('.pt')]
    # get all blob names
    blob_names = [f[:-3] for f in os.listdir(path_to_data + "/") if f.endswith('.pt')]  # remove .pt ending
    # select only how many blobs should be used for training
    blobs_paths = blobs_paths[:number_of_blobs]
    blob_names = blob_names[:number_of_blobs]
    return blobs_paths, blob_names


def get_total_number_of_samples(blob_names: list, train_lenghts: dict, val_lenghts: dict, test_lengths: dict) -> tuple:
    """
    Retrieves the total number of training, validation and testing samples for a given number of blobs (coming from the same directory).

    Args:
        blob_names (list): List containing the names of the blobs.
        train_lenghts (dict): Dictionary containing the number of training samples for each blob.
        val_lenghts (dict): Dictionary containing the number of validation samples for each blob.
        test_lengths (dict): Dictionary containing the number of testing samples for each blob.

    Returns:
        tuple: Tuple containing the total number of training, validation and testing samples for the specified number of blobs.
    """
    train_lenghts = {key: train_lenghts[key] for key in blob_names}
    val_lenghts = {key: val_lenghts[key] for key in blob_names}
    test_lengths = {key: test_lengths[key] for key in blob_names}
    # get total number of training, validation and testing samples
    total_train_len = sum(train_lenghts.values())
    total_val_len = sum(val_lenghts.values())
    total_test_len = sum(test_lengths.values())
    return total_train_len, total_val_len, total_test_len


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


