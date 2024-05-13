import os

import torch
from tqdm import tqdm

"""
screen -dmS remove_artefact_files sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/remove_artefact_files.py; exec bash'
"""


def main():
    # embeddings_normalized_patches
    slides = os.listdir("/data/pamly/embeddings_normalized_patches")
    count = 0
    for slide in tqdm(slides, total=len(slides)):

        patches = os.listdir(f"/data/pamly/embeddings_normalized_patches/{slide}")
        for patch in tqdm(patches, total=len(patches), desc=f"Removing Artifact Patches from slide {slide}"):
            image, label = torch.load(f"/data/pamly/embeddings_normalized_patches/{slide}/{patch}")

            if label == 1:
                count += 1
                os.remove(f"/data/pamly/embeddings_normalized_patches/{slide}/{patch}")
    print(f"Removed {count} patches with label 1.")


if __name__ == "__main__":
    main()
