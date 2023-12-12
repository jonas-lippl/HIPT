import argparse
import os

import torch
from torchvision import models as torchvision_models
from torchvision import transforms
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
    teacher = vits.vit_small(patch_size=16)
    state_dict = torch.load("/mnt/ckpts/pretrain_40_epochs_64_bs/checkpoint0039.pth", map_location="cpu")['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = teacher.load_state_dict(state_dict, strict=False)

    teacher.eval()
    teacher.to(device)

    patch_dir = "/data/256x384_embedding_tokens"
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir, exist_ok=True)
    count = 0

    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    patches = [patch for patch in os.listdir("/data/single_4096_px_2048mu") if patch.startswith("patch")]
    with torch.no_grad():
        for patch in tqdm(patches[12000:]):
            name = patch.split(".")[0]
            file = os.path.join(patch_dir, f"{name}_256x384_embedding.pt")
            if os.path.exists(file):
                continue
            img_4096, _ = torch.load(os.path.join("/data/single_4096_px_2048mu", patch))
            batch = torch.zeros((256, 3, 256, 256))
            for i in range(16):
                for j in range(16):
                    batch[i * 16 + j] = transform(img_4096[:, i * 256: (i + 1) * 256, j * 256:(j + 1) * 256].clone().div(255.0))
            out = teacher(batch.to(device)).to('cpu')
            torch.save(out, file)
            count += 1
            if count % 100 == 0:
                print(f"Saved {count} patch embeddings.")
        print(f"Saved {count} patch embeddings.")


if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
