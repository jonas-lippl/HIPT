import os

import torch
from torchvision import transforms
from tqdm import tqdm

"""
screen -dmS normalize_patches sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/normalize_slide_patches.py; exec bash'
"""


def main():
    slides = os.listdir("/data/pamly/sampled_tiles_labeled_only")
    for slide in tqdm(slides, total=len(slides)):
        if not os.path.exists(f"/data/pamly/slide_means_stds_labeled_only/{slide}"):
            os.makedirs(f"/data/pamly/slide_means_stds_labeled_only/{slide}", exist_ok=True)

        all_images = []
        patches = os.listdir(f"/data/pamly/sampled_tiles_labeled_only/{slide}")
        for patch in tqdm(patches, total=len(patches), desc=f"Calculating mean and std for slide {slide}"):
            image, label = torch.load(f"/data/pamly/sampled_tiles_labeled_only/{slide}/{patch}")

            all_images.append(image.float().div(255))
        stacked_images = torch.stack(all_images)
        print(stacked_images.shape)
        mean = stacked_images.mean(dim=[0, 2, 3])
        std = stacked_images.std(dim=[0, 2, 3])
        print(f"Slide {slide} before Normalization: Mean {mean}, Std {std}")

        torch.save((mean, std), f"/data/pamly/slide_means_stds_labeled_only/{slide}/mean_std.pt")


if __name__ == '__main__':
    main()
