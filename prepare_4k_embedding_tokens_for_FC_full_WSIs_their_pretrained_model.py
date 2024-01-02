import argparse
import os
import re

import torch
from pamly import Diagnosis
from torchvision import models as torchvision_models
from torchvision import transforms
from tqdm import tqdm

from HIPT_4K.hipt_4k import HIPT_4K

"""
screen -dmS generate_patch_embeddings_0 sh -c 'docker run --gpus \"device=0\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_their_pretrained_model.py --start 0 --stop 33; exec bash'
screen -dmS generate_patch_embeddings_1 sh -c 'docker run --gpus \"device=1\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_their_pretrained_model.py --start 33 --stop 66; exec bash'
screen -dmS generate_patch_embeddings_2 sh -c 'docker run --gpus \"device=2\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_their_pretrained_model.py --start 66 --stop 99; exec bash'
screen -dmS generate_patch_embeddings_3 sh -c 'docker run --gpus \"device=3\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_their_pretrained_model.py --start 99 --stop 132; exec bash'
screen -dmS generate_patch_embeddings_4 sh -c 'docker run --gpus \"device=4\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_their_pretrained_model.py --start 132 --stop 165; exec bash'
screen -dmS generate_patch_embeddings_5 sh -c 'docker run --gpus \"device=5\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_their_pretrained_model.py --start 165 --stop 198; exec bash'
screen -dmS generate_patch_embeddings_6 sh -c 'docker run --gpus \"device=6\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_their_pretrained_model.py --start 198 --stop 231; exec bash'
screen -dmS generate_patch_embeddings_7 sh -c 'docker run --gpus \"device=7\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC_full_WSIs_their_pretrained_model.py --start 231 --stop 264; exec bash'
"""

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

LABELS_MAP = {
    "Unknown": int(Diagnosis("Unknown")),  # for unknown diagnosis
    "HL": int(Diagnosis("HL")),  # Hodgkin Lymphoma
    "DLBCL": int(Diagnosis("DLBCL")),  # Diffuse Large B-Cell Lymphoma
    "CLL": int(Diagnosis("CLL")),  # Chronic Lymphocytic Leukemia
    "FL": int(Diagnosis("FL")),  # Follicular Lymphoma
    "MCL": int(Diagnosis("MCL")),  # Mantle Cell Lymphoma
    "LYM": int(Diagnosis("LTDS")),  # Lymphadenitis
    "Lts": int(Diagnosis("LTDS")),  # Lymphadenitis
}

INT2STR_LABEL_MAP = {
    int(Diagnosis("Unknown")): "Unknown",
    int(Diagnosis("HL")): "HL",
    int(Diagnosis("DLBCL")): "DLBCL",
    int(Diagnosis("CLL")): "CLL",
    int(Diagnosis("FL")): "FL",
    int(Diagnosis("MCL")): "MCL",
    int(Diagnosis("LTDS")): "Lts",
}


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--start', default=0, type=int, help='Patch resolution of the model.')
    parser.add_argument('--stop', default=200, type=int, help='Patch resolution of the model.')

    return parser


def extract_coordinates(string):
    # Find all occurrences of numbers in the string
    numbers = re.findall(r'\d+', string)

    # Convert the list of strings to a list of integers
    numbers = [int(number) for number in numbers]

    # Return only the second and third numbers (456 and 789)
    return numbers[1:3]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HIPT_4K(device256=device, device4k=device)

    count = 0

    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # For wsis with 0 overlap between 4k patches:
    wsi_dir = "/data/WSI_patches_4096px_2048mu"
    # wsi_embedding_dir = "/data/WSI_patches_4096px_2048mu_4k_embeddings"
    wsi_embedding_dir = "/data/WSI_patches_4096px_2048mu_4k_embeddings_their_pretrained_model"
    # For wsis with 0.5 overlap between 4k patches:
    # wsi_dir = "/data/WSI_patches_4096px_2048mu_0.5overlap"
    # wsi_embedding_dir = "/data/WSI_patches_4096px_2048mu_0.5overlap_4k_embeddings"
    wsis = sorted(os.listdir(wsi_dir))
    with torch.no_grad():
        for wsi in tqdm(wsis[args.start:args.stop]):
            os.makedirs(wsi_embedding_dir, exist_ok=True)
            if os.path.exists(os.path.join(wsi_embedding_dir, f"{wsi}.pt")):
                continue
            patches = os.listdir(os.path.join(wsi_dir, wsi))
            print(f"Processing {wsi} with {len(patches)} patches.")
            num_patches = len(patches)
            if num_patches == 0:
                print(f"Skipping {wsi} because it has no patches.")
                continue
            embeddings = torch.zeros((num_patches, 192))
            label = torch.tensor(LABELS_MAP[wsi.split("-")[-1]])
            for k, patch in enumerate(patches):
                img_4096 = torch.load(os.path.join(os.path.join(wsi_dir, wsi), patch))
                out = model(transform(img_4096.clone().div(255.0)).unsqueeze(dim=0)).squeeze(dim=0).cpu()
                embeddings[k] = out
            torch.save((embeddings, label), os.path.join(wsi_embedding_dir, f"{wsi}.pt"))
            count += 1
            if count % 10 == 0:
                print(f"Saved {count} patch 4k embeddings.")
        print(f"Saved {count} patch embeddings in total.")


if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
