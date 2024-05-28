import os
import torch
from tqdm import tqdm

"""
screen -dmS decrease_uni_embeddings sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job6 -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data/pamly/embeddings:/embeddings -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/decrease_uni_embedding_storage.py; exec bash'
screen -dmS decrease_uni_embeddings sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job7 -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data/pamly/embeddings_512px:/embeddings -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/decrease_uni_embedding_storage.py; exec bash'
screen -dmS decrease_uni_embeddings sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job8 -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data/pamly/embeddings_512px_labeled_only:/embeddings -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/decrease_uni_embedding_storage.py; exec bash'
screen -dmS decrease_uni_embeddings sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job9 -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data/pamly/embeddings_512px_normalized_patches:/embeddings -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/decrease_uni_embedding_storage.py; exec bash'
screen -dmS decrease_uni_embeddings sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job10 -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data/pamly/embeddings_512px_normalized_patches_labeled_only:/embeddings -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/decrease_uni_embedding_storage.py; exec bash'
screen -dmS decrease_uni_embeddings sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job11 -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data/pamly/embeddings_normalized_patches:/embeddings -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/decrease_uni_embedding_storage.py; exec bash'
screen -dmS decrease_uni_embeddings sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job12 -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data/pamly/embeddings_normalized_patches_labeled_only:/embeddings -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/decrease_uni_embedding_storage.py; exec bash'
screen -dmS decrease_uni_embeddings sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job13 -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data/pamly/embeddings_labeled_only:/embeddings -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/decrease_uni_embedding_storage.py; exec bash'
"""


if __name__ == '__main__':
    embedding_dir = "/embeddings"

    for slide in tqdm(os.listdir(embedding_dir)):
        print(slide)
        for patch in tqdm(os.listdir(f"{embedding_dir}/{slide}")):
            embedding, label = torch.load(f"{embedding_dir}/{slide}/{patch}", map_location="cpu")
            embedding = embedding.detach().cpu()
            label = label.detach().cpu()
            torch.save((embedding, label), f"{embedding_dir}/{slide}/{patch}")
