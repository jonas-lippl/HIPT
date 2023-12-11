import argparse
import os

import torch
from torchvision import models as torchvision_models
from tqdm import tqdm

import utils
import vision_transformer as vits

"""
screen -dmS generate_patch_embeddings sh -c 'docker run --gpus \"device=1\" -it -u `id -u $USER` --rm -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens.py; exec bash'
"""

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')

    return parser


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = vits.__dict__[args.arch](patch_size=args.patch_size)

    utils.restart_from_checkpoint("/mnt/ckpts/pretrain_40_epochs_64_bs/checkpoint0006.pth", teacher=teacher)
    teacher.eval()
    teacher.to(device)

    patch_dir = "/data/256x384_embedding_tokens"
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir, exist_ok=True)
    count = 0

    for patch in tqdm(os.listdir("/data/single_4096_px_2048mu")):
        if not patch.startswith("patch"):
            name = patch.split(".")[0]
            img_4096, _ = torch.load(os.path.join("/data/single_4096_px_2048mu", patch))
            embedding = torch.zeros((256, 384))
            for i in range(16):
                for j in range(16):
                    img = img_4096[:, i * 256: (i + 1) * 256, j * 256:(j + 1) * 256].clone()
                    out = teacher(img.div(255).unsqueeze(0).to(device)).squeeze(0).to('cpu')
                    print(out.shape)
                    embedding[i * 16 + j] = out
            print("Tensor shape: ", embedding.shape)
            torch.save(embedding, os.path.join(patch_dir, f"{name}_256x384_embedding.pt"))
            count += 1
            if count == 10:
                break


if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
