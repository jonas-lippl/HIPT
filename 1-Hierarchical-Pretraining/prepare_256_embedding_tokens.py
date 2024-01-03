import argparse
import os

import torch
from torchvision import models as torchvision_models
from torchvision import transforms
from tqdm import tqdm

import vision_transformer as vits

"""
screen -dmS generate_patch_embeddings0 sh -c 'docker run --gpus \"device=0\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens.py --start=0 --stop=0.125; exec bash'
screen -dmS generate_patch_embeddings1 sh -c 'docker run --gpus \"device=1\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens.py --start=0.125 --stop=0.25; exec bash'
screen -dmS generate_patch_embeddings2 sh -c 'docker run --gpus \"device=2\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens.py --start=0.25 --stop=0.375; exec bash'
screen -dmS generate_patch_embeddings3 sh -c 'docker run --gpus \"device=3\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens.py --start=0.375 --stop=0.5; exec bash'
screen -dmS generate_patch_embeddings4 sh -c 'docker run --gpus \"device=4\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens.py --start=0.5 --stop=0.625; exec bash'
screen -dmS generate_patch_embeddings5 sh -c 'docker run --gpus \"device=5\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens.py --start=0.625 --stop=0.75; exec bash'
screen -dmS generate_patch_embeddings6 sh -c 'docker run --gpus \"device=6\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens.py --start=0.75 --stop=0.875; exec bash'
screen -dmS generate_patch_embeddings7 sh -c 'docker run --gpus \"device=7\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens.py --start=0.875 --stop=1.0; exec bash'
"""

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('Patch Embeddings', add_help=False)
    parser.add_argument('--start', default=0.0, type=float, help='Patch resolution of the model.')
    parser.add_argument('--stop', default=1.0, type=float, help='Patch resolution of the model.')
    return parser


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vits.vit_small(patch_size=16)
    state_dict = torch.load("/mnt/ckpts/pretrain_40_epochs_64_bs_resnet/checkpoint.pth", map_location="cpu")['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    save_dir = "/data/256x384_embedding_tokens_resnet"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    count = 0

    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # wsis = [wsi for wsi in os.listdir("/data/WSI_patches_4096px_2048mu")]
    # for wsi in tqdm(wsis):
    #   patch_dir = "/data/WSI_patches_4096px_2048mu/"+wsi
    patch_dir = "/data/single_4096_px_2048mu"
    num_patches = len(os.listdir(patch_dir))
    patches = os.listdir(patch_dir)[int(num_patches * args.start): int(num_patches * args.stop)]
    with torch.no_grad():
        for patch in tqdm(patches):
            name = patch.split(".")[0]
            file = os.path.join(save_dir, f"{name}_256x384_embedding.pt")
            if os.path.exists(file):
                print(f"Skipping {file}")
                continue
            img_4096, _ = torch.load(os.path.join(patch_dir, patch))
            batch = torch.zeros((256, 3, 256, 256))
            for i in range(16):
                for j in range(16):
                    batch[i * 16 + j] = transform(
                        img_4096[:, i * 256: (i + 1) * 256, j * 256:(j + 1) * 256].clone().div(255.0))
            out = model(batch.to(device)).to('cpu')
            torch.save(out, file)
            count += 1
        if count % 100 == 0:
            print(f"Saved {count} patch embeddings.")
    print(f"Saved {count} patch embeddings.")


if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
