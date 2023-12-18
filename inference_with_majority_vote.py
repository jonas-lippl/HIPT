

from pamly import Diagnosis
from torch.backends import cudnn

import torch

from HIPT_4K.hipt_4k import ClassificationHead
from utils.load_data import load_lymphoma_data_WSI_embeddings

"""
screen -dmS hipt sh -c 'docker run --shm-size=400gb --gpus all  -it --rm -u `id -u $USER` -v /sybig/home/jol/Code/blobyfire/data/WSI_patches_4096px_2048mu_0.5overlap_4k_embeddings:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/inference_with_majority_vote.py; exec bash'
"""

INT2STR_LABEL_MAP = {
    int(Diagnosis("Unknown")): "Unknown",
    int(Diagnosis("HL")): "HL",
    int(Diagnosis("DLBCL")): "DLBCL",
    int(Diagnosis("CLL")): "CLL",
    int(Diagnosis("FL")): "FL",
    int(Diagnosis("MCL")): "MCL",
    int(Diagnosis("LTDS")): "Lts",
}


def general_setup(seed: int = 1, benchmark=True, hub_dir: str = 'tmp/'):
    """
    General setup for training.
    """
    # en- or disable cudnn auto-tuner to find the best algorithm for current hardware
    cudnn.benchmark = benchmark
    torch.hub.set_dir(hub_dir)
    torch.manual_seed(seed)


def main():
    general_setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = load_lymphoma_data_WSI_embeddings()
    classifier = ClassificationHead().to(device)
    classifier.load_state_dict(torch.load("/mnt/experiments/hipt_4k_extra_data/classifier.pt"))

    with torch.no_grad():
        correct = 0
        per_class_count = {i: 0 for i in range(1, 7)}
        per_class_correct = {i: 0 for i in range(1, 7)}
        single_patch_per_wsi_counter = 0
        for X, y in data_loader:
            X = X.to(device).squeeze(dim=0)
            y = y.to(device).squeeze(dim=0)
            print("X shape: ", X.shape)
            # if X.shape[0] <= 10:
            #     print(f"Skipping WSI {name} with only {X.shape[0]} patches")
            #     continue
            prob, pred = classifier.forward(X)
            print("Predictions: ", pred)
            per_class_pred = {i: 0 for i in range(7)}
            try:
                for i in pred:
                    per_class_pred[i.item()] += 1
            except TypeError:
                single_patch_per_wsi_counter += 1
                per_class_pred[pred.item()] += 1
            print("Per class predictions: ", per_class_pred)
            majority_vote = max(per_class_pred, key=per_class_pred.get)
            print("Majority vote: ", majority_vote)
            print("True label: ", y.item())
            per_class_count[y.item()] += 1
            if majority_vote == y.item():
                correct += 1
                per_class_correct[y.item()] += 1
        print("Total Accuracy: ", correct / sum(per_class_count.values()))
        for k in range(1, 7):
            print(f"Class {k} ({INT2STR_LABEL_MAP[k]}): {per_class_correct[k]} / {per_class_count[k]} = {per_class_correct[k] / per_class_count[k]}")

        print(f"For {single_patch_per_wsi_counter} WSIs only one 4k patch was available")


if __name__ == '__main__':
    main()
