import argparse
import os

import torch
from pamly import Diagnosis
from torchvision import transforms
from tqdm import tqdm

from HIPT_4K.hipt_4k import HIPT_4K

"""
screen -dmS generate_patch_embeddings_0 sh -c 'docker run --gpus \"device=0\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC.py --start 0.0 --stop 0.15; exec bash'
screen -dmS generate_patch_embeddings_1 sh -c 'docker run --gpus \"device=1\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC.py --start 0.15 --stop 0.30; exec bash'
screen -dmS generate_patch_embeddings_2 sh -c 'docker run --gpus \"device=2\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC.py --start 0.30 --stop 0.45; exec bash'
screen -dmS generate_patch_embeddings_3 sh -c 'docker run --gpus \"device=3\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC.py --start 0.45 --stop 0.60; exec bash'
screen -dmS generate_patch_embeddings_4 sh -c 'docker run --gpus \"device=4\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC.py --start 0.60 --stop 0.75; exec bash'
screen -dmS generate_patch_embeddings_5 sh -c 'docker run --gpus \"device=5\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC.py --start 0.75 --stop 0.90; exec bash'
screen -dmS generate_patch_embeddings_6 sh -c 'docker run --gpus \"device=6\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens_for_FC.py --start 0.90 --stop 1.0; exec bash'
"""

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


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    parser.add_argument('--start', default=0.0, type=float, help='Patch resolution of the model.')
    parser.add_argument('--stop', default=1.0, type=float, help='Patch resolution of the model.')

    return parser


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HIPT_4K(device256=device, device4k=device)

    count = 0

    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # embedding_dir = "/data/single_4096px_2048mu_embeddings_their_model_test"
    # patch_dir = "/data/single_4096px_2048mu_test"
    embedding_dir = "/data/single_4096px_2048mu_embeddings_their_model_train"
    patch_dir = "/data/single_4096px_2048mu_train"

    patches = [patch for patch in os.listdir(patch_dir)]
    total = len(patches)
    print(f"Found {len(patches)} patches.")
    os.makedirs(embedding_dir, exist_ok=True)
    with torch.no_grad():
        for patch in tqdm(patches[int(total*args.start):int(total*args.stop)]):
            img, label = torch.load(os.path.join(patch_dir, patch))
            out = model(transform(img.clone().div(255.0)).unsqueeze(dim=0)).squeeze(dim=0).cpu()
            torch.save((out, label), os.path.join(embedding_dir, f"{patch}.pt"))
            count += 1
            if count % 100 == 0:
                print(f"Saved {count} patch 4k embeddings.")
        print(f"Saved {count} patch embeddings in total.")


if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
