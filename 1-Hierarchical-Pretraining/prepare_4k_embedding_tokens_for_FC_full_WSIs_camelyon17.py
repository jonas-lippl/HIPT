import argparse
import os

import torch
from torchvision import models as torchvision_models
from tqdm import tqdm
import pandas as pd

import vision_transformer4k as vits4k

"""
screen -dmS generate_patch_embeddings_0 sh -c 'docker run --gpus \"device=0\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_camelyon17.py --center=0; exec bash'
screen -dmS generate_patch_embeddings_1 sh -c 'docker run --gpus \"device=1\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_camelyon17.py --center=1; exec bash'
screen -dmS generate_patch_embeddings_2 sh -c 'docker run --gpus \"device=2\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_camelyon17.py --center=2; exec bash'
screen -dmS generate_patch_embeddings_3 sh -c 'docker run --gpus \"device=3\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_camelyon17.py --center=3; exec bash'
screen -dmS generate_patch_embeddings_4 sh -c 'docker run --gpus \"device=4\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_camelyon17.py --center=4; exec bash'
"""

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

LABELS_MAP = {
    'negative': 0,
    'itc': 1,
    'micro': 2,
    'macro': 3,
}


def get_args_parser():
    parser = argparse.ArgumentParser('Camelyon17 WSI embeddings', add_help=False)
    parser.add_argument('--center', default=0, type=int, help='Number of the center to process.')
    return parser


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vit4k = vits4k.vit4k_xs(patch_size=16)
    state_dict = torch.load("/mnt/ckpts/pretrain4k_100_epochs_64_bs_camelyon17/checkpoint.pth", map_location="cpu")['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = vit4k.load_state_dict(state_dict, strict=False)

    vit4k.eval()
    vit4k.to(device)

    count = 0
    embedding_dir = f"/data/patches/patches_{args.center}_4096_lvl_1/256x384_embedding_tokens"
    wsi_embedding_dir = f"/data/patches/patches_{args.center}_4096_lvl_1/4k_embedding_tokens"

    stage_labels = pd.read_csv("/data/stage_labels.csv")
    if not os.path.exists(wsi_embedding_dir):
        os.makedirs(wsi_embedding_dir, exist_ok=True)
    with torch.no_grad():
        for patient in tqdm(os.listdir(embedding_dir)):
            patches = os.listdir(os.path.join(embedding_dir, patient))
            patches.sort()
            print(f"Processing {patient} with {len(patches)} patches.")
            num_patches = len(patches)
            if num_patches == 0:
                print(f"Skipping {patient} because it has no patches.")
                continue
            embeddings = torch.zeros((num_patches, 192))
            stage = stage_labels[stage_labels["patient"] == patient+'.tif']["stage"].values[0]
            label = torch.tensor(LABELS_MAP[stage])
            for k, patch in enumerate(patches):
                embedding = torch.load(os.path.join(embedding_dir, patient, patch))
                embedding = embedding.unfold(0, 16, 16).transpose(0,1).to(device)
                out = vit4k(embedding.unsqueeze(dim=0)).squeeze(dim=0).cpu()
                embeddings[k] = out
            torch.save((embeddings, label), os.path.join(wsi_embedding_dir, f"{patient}.pt"))
            count += 1
            if count % 10 == 0:
                print(f"Saved {count} patch 4k embeddings.")
        print(f"Saved {count} patch embeddings in total.")


if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    parser = argparse.ArgumentParser('Camelyon17 WSI embeddings', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
