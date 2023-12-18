import argparse
import os

import torch
from pamly import Diagnosis
from torchvision import models as torchvision_models
from torchvision import transforms
from tqdm import tqdm

import utils
import vision_transformer as vits
import vision_transformer4k as vits4k
import pamly

"""
screen -dmS generate_patch_embeddings_0 sh -c 'docker run --gpus \"device=0\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 0 --stop 33; exec bash'
screen -dmS generate_patch_embeddings_1 sh -c 'docker run --gpus \"device=1\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 33 --stop 66; exec bash'
screen -dmS generate_patch_embeddings_2 sh -c 'docker run --gpus \"device=2\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 66 --stop 99; exec bash'
screen -dmS generate_patch_embeddings_3 sh -c 'docker run --gpus \"device=3\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 99 --stop 132; exec bash'
screen -dmS generate_patch_embeddings_4 sh -c 'docker run --gpus \"device=4\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 132 --stop 165; exec bash'
screen -dmS generate_patch_embeddings_5 sh -c 'docker run --gpus \"device=5\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 165 --stop 198; exec bash'
screen -dmS generate_patch_embeddings_6 sh -c 'docker run --gpus \"device=6\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 198 --stop 231; exec bash'
screen -dmS generate_patch_embeddings_7 sh -c 'docker run --gpus \"device=7\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_4k_embedding_tokens.py --start 231 --stop 264; exec bash'
"""

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

LABELS_MAP = {
    "Unknown": int(Diagnosis("Unknown")),      # for unknown diagnosis
    "HL":      int(Diagnosis("HL")),           # Hodgkin Lymphoma
    "DLBCL":   int(Diagnosis("DLBCL")),        # Diffuse Large B-Cell Lymphoma
    "CLL":     int(Diagnosis("CLL")),          # Chronic Lymphocytic Leukemia
    "FL":      int(Diagnosis("FL")),           # Follicular Lymphoma
    "MCL":     int(Diagnosis("MCL")),          # Mantle Cell Lymphoma
    "LYM":     int(Diagnosis("LTDS")),         # Lymphadenitis
    "Lts":     int(Diagnosis("LTDS")),         # Lymphadenitis
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
    state_dict = torch.load("/mnt/ckpts/pretrain4k_100_epochs_64_bs/checkpoint.pth", map_location="cpu")['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = vit4k.load_state_dict(state_dict, strict=False)

    vit4k.eval()
    vit4k.to(device)

    count = 0

    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # For wsis with 0 overlap between 4k patches:
    # wsi_dir = "/data/WSI_patches_4096px_2048mu"
    # wsi_embedding_dir = "/data/WSI_patches_4096px_2048mu_4k_embeddings"
    # For wsis with 0.5 overlap between 4k patches:
    wsi_dir = "/data/WSI_patches_4096px_2048mu_0.5overlap"
    wsi_embedding_dir = "/data/WSI_patches_4096px_2048mu_0.5overlap_4k_embeddings"
    wsis = [wsi for wsi in os.listdir(wsi_dir)]
    with torch.no_grad():
        for wsi in tqdm(wsis[args.start:args.stop]):
            os.makedirs(wsi_embedding_dir, exist_ok=True)
            if os.path.exists(os.path.join(wsi_embedding_dir, f"{wsi}.pt")):
                continue
            patches = os.listdir(os.path.join(wsi_dir, wsi))
            print(f"Processing {wsi} with {len(patches)} patches.")
            num_patches = len(patches)
            embeddings = torch.zeros((num_patches, 192))
            label = torch.tensor(LABELS_MAP[wsi.split("-")[-1]])
            for k, patch in enumerate(patches):
                img_4096 = torch.load(os.path.join(os.path.join(wsi_dir, wsi), patch))
                batch = torch.zeros((256, 3, 256, 256))
                for i in range(16):
                    for j in range(16):
                        batch[i * 16 + j] = transform(img_4096[:, i * 256: (i + 1) * 256, j * 256:(j + 1) * 256].clone().div(255.0))
                out = vit256(batch.to(device))
                out = out.unfold(0, 16, 16).transpose(0,1)
                out = vit4k(out.unsqueeze(dim=0)).squeeze(dim=0).cpu()
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
