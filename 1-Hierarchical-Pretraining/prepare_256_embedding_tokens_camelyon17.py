import argparse
import os

import torch
from torchvision import models as torchvision_models
from torchvision import transforms
from tqdm import tqdm

import vision_transformer as vits

"""
screen -dmS generate_patch_embeddings0 sh -c 'docker run --gpus \"device=0\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17/patches_center_0_4k:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens_camelyon17.py; exec bash'
screen -dmS generate_patch_embeddings1 sh -c 'docker run --gpus \"device=1\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17/patches_center_1_4k:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens_camelyon17.py; exec bash'
screen -dmS generate_patch_embeddings2 sh -c 'docker run --gpus \"device=2\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17/patches_center_2_4k:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens_camelyon17.py; exec bash'
screen -dmS generate_patch_embeddings3 sh -c 'docker run --gpus \"device=3\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17/patches_center_3_4k:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens_camelyon17.py; exec bash'
screen -dmS generate_patch_embeddings4 sh -c 'docker run --gpus \"device=4\" -it -u `id -u $USER` --rm -v /sybig/projects/camelyon17/patches_center_4_4k:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt python3 /mnt/prepare_256_embedding_tokens_camelyon17.py; exec bash'
"""

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vits.vit_small(patch_size=16)
    state_dict = torch.load("/mnt/ckpts/pretrain_100_epochs_64_bs_vit/checkpoint.pth", map_location="cpu")['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    model.eval()
    model.to(device)
    count = 0
    with torch.no_grad():
        for patient_node in tqdm(os.listdir("/data")):
            if 'embedding_tokens' in patient_node:
                continue
            save_dir = os.path.join("/data", "256x384_embedding_tokens", patient_node)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            for patch_4k in os.listdir(os.path.join("/data", patient_node)):
                name = patch_4k.split(".")[0]
                file = os.path.join(save_dir, f"{name}_embedding.pt")
                # if os.path.exists(file):
                #     print(f"Skipping {file}")
                #     continue
                img_4096 = torch.load(os.path.join("/data", patient_node, patch_4k))
                batch = torch.zeros((256, 3, 256, 256))
                for i in range(16):
                    for j in range(16):
                        batch[i * 16 + j] = img_4096[:, i * 256: (i + 1) * 256, j * 256:(j + 1) * 256].clone()
                out = model(batch.to(device)).to('cpu')
                torch.save(out, file)
                count += 1
                if count % 100 == 0:
                    print(f"Saved {count} patch embeddings.")
        print(f"Saved {count} patch embeddings.")


if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    main()
