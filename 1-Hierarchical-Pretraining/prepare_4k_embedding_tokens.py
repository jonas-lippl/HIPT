import argparse
import os

import torch
from pamly import Diagnosis
from torchvision import models as torchvision_models
from torchvision import transforms
from tqdm import tqdm

import vision_transformer as vits
import vision_transformer4k as vits4k

"""
screen -dmS generate_patch_embeddings_0 sh -c 'docker run --gpus \"device=0\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 0.0 --stop 0.125; exec bash'
screen -dmS generate_patch_embeddings_1 sh -c 'docker run --gpus \"device=1\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 0.125 --stop 0.25; exec bash'
screen -dmS generate_patch_embeddings_2 sh -c 'docker run --gpus \"device=2\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 0.25 --stop 0.375; exec bash'
screen -dmS generate_patch_embeddings_3 sh -c 'docker run --gpus \"device=3\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 0.375 --stop 0.5; exec bash'
screen -dmS generate_patch_embeddings_4 sh -c 'docker run --gpus \"device=4\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 0.5 --stop 0.625; exec bash'
screen -dmS generate_patch_embeddings_5 sh -c 'docker run --gpus \"device=5\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 0.625 --stop 0.75; exec bash'
screen -dmS generate_patch_embeddings_6 sh -c 'docker run --gpus \"device=6\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 0.75 --stop 0.875; exec bash'
screen -dmS generate_patch_embeddings_7 sh -c 'docker run --gpus \"device=7\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 0.875 --stop 1.0; exec bash'
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
    vit256 = vits.vit_small(patch_size=16)
    state_dict = torch.load("/mnt/ckpts/pretrain_40_epochs_64_bs/checkpoint.pth", map_location="cpu")['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = vit256.load_state_dict(state_dict, strict=False)

    vit256.eval()
    vit256.to(device)

    vit4k = vits4k.vit4k_xs(patch_size=16)
    state_dict = torch.load("/mnt/ckpts/pretrain4k_100_epochs_64_bs_additional_data/checkpoint.pth", map_location="cpu")['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = vit4k.load_state_dict(state_dict, strict=False)

    vit4k.eval()
    vit4k.to(device)

    count = 0

    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    patches = [patch for patch in os.listdir("/data/single_4096_px_2048mu_train")]
    total = len(patches)
    print(f"Found {total} patches.")
    os.makedirs("/data/single_4096px_2048mu_embeddings_train", exist_ok=True)
    with torch.no_grad():
        for patch in patches[total * args.start: total * args.stop]:
            if os.path.exists(os.path.join("/data/single_4096px_2048mu_embeddings_train", f"{patch}.pt")):
                continue
            img, label = torch.load(os.path.join("/data/single_4096_px_2048mu_train", patch))
            batch = torch.zeros((256, 3, 256, 256))
            for i in range(16):
                for j in range(16):
                    batch[i * 16 + j] = transform(
                        img[:, i * 256: (i + 1) * 256, j * 256:(j + 1) * 256].clone().div(255.0))
            out = vit256(batch.to(device))
            out = out.unfold(0, 16, 16).transpose(0, 1)
            out = vit4k(out.unsqueeze(dim=0)).squeeze(dim=0).cpu()
            torch.save((out, label), os.path.join("/data/single_4096px_2048mu_embeddings_train", f"{patch}.pt"))
            count += 1
            if count % 100 == 0:
                print(f"Saved {count} patch 4k embeddings.")
        print(f"Saved {count} patch embeddings in total.")


if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
