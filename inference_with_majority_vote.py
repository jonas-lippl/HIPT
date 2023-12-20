import os
import re

import torchvision.transforms
from matplotlib import pyplot as plt, patches
from pamly import Diagnosis
from torch.backends import cudnn

import torch
from torchvision import transforms

from HIPT_4K.hipt_4k import ClassificationHead
from generate_overlays import get_limits_of_plot, add_labels_and_save_plot, prepare_transformed_tensor_for_plotting
from utils.load_data import load_lymphoma_data_WSI_embeddings

"""
screen -dmS hipt_WSI_patches sh -c 'docker run --shm-size=400gb --gpus all  -it --rm -u `id -u $USER` -v /sybig/home/jol/Code/blobyfire/data/WSI_patches_4096px_2048mu_4k_embeddings:/data -v /sybig/home/jol/Code/blobyfire/data/WSI_patches_4096px_2048mu:/original_patches -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/inference_with_majority_vote.py; exec bash'
screen -dmS hipt_WSI_patches_overlap sh -c 'docker run --shm-size=400gb --gpus all  -it --rm -u `id -u $USER` -v /sybig/home/jol/Code/blobyfire/data/WSI_patches_4096px_2048mu_0.5overlap_4k_embeddings:/data -v /sybig/home/jol/Code/blobyfire/data/WSI_patches_4096px_2048mu:/original_patches -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/inference_with_majority_vote.py; exec bash'
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


def extract_coordinates(string):
    # Find all occurrences of numbers in the string
    numbers = re.findall(r'\d+', string)
    # Convert the list of strings to a list of integers
    numbers = [int(number) for number in numbers]
    # Return only the second and third numbers (456 and 789)
    return numbers[1:3]


def plot_patches(images, coordinates: list, patch_size: int, correct_diagnosis: int, ax: plt.Axes, pred) -> tuple[plt.Axes, list[str]]:
    # create list of colors for the different classes
    colors = ['green', 'blue', 'orange', 'purple', 'brown', 'yellow']

    # iterate over all patches and plot them on the empty plot
    for i, x in enumerate(images):
        coord = coordinates[i]

        if x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        ax.imshow(x, extent=[coord[0], coord[0] + patch_size, coord[1], coord[1] - patch_size])

        # add colored square on top of the patch according to the prediction
        rectangle = patches.Rectangle((coord[0], coord[1] - patch_size), patch_size, patch_size, edgecolor='none',
                                      facecolor=colors[pred[0]], alpha=0.4)
        ax.add_patch(rectangle)

        print(f"Processed patches for class {correct_diagnosis}: {i + 1}/{len(images)}", end="\r")

    print()

    return ax, colors


def plot_overlay(name, pred, y):
    name = name[0].replace(".pt", "")
    tile_size = 512
    coordinates = []
    patches = []
    for file in os.listdir(f"/original_patches/{name}"):
        coordinates.append(tuple(extract_coordinates(file)))  # form  [(20, 40), (21, 35), ...]
        patches.append(torch.load(f"/original_patches/{name}/{file}"))
    coordinates = [(coord[0] * tile_size, coord[1] * tile_size) for coord in coordinates]
    x_min, x_max, y_min, y_max = get_limits_of_plot(coordinates)

    # create empty plot on which we place the patches where they belong using the coordinates
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')

    ax, colors = plot_patches(patches, coordinates, 4096, y, ax, pred)

    add_labels_and_save_plot(ax, colors, y, f"/mnt/experiments/inference_with_majority_vote", "wrong_patches", name, dpi=500)


def plot_wrong_patches(X, name, pred, y):
    disagree_indices = torch.nonzero(pred != y, as_tuple=True)
    disagree_indices_list = [index.item() for index in disagree_indices[0]]
    name = name[0]
    print("name: ", name)
    name = name.replace(".pt", "")
    for index in disagree_indices_list[:5]:
        file = os.listdir(f"/original_patches/{name}")[index]
        print(file)
        patch = torch.load(f"/original_patches/{name}/{file}")
        if patch.shape[0] == 3:
            patch = patch.permute(1, 2, 0)
        plt.imshow(patch.cpu().numpy())
        plt.title(f'Patch {index} of {name}, Predicted: {pred[index]}, Actual: {y.item()}')
        if not os.path.exists(f"/mnt/experiments/inference_with_majority_vote/wrong_patches/{name}"):
            os.makedirs(f"/mnt/experiments/inference_with_majority_vote/wrong_patches/{name}", exist_ok=True)
        plt.savefig(f"/mnt/experiments/inference_with_majority_vote/wrong_patches/{name}/example_{index}.png", dpi=500)
        plt.close()


def main():
    general_setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = load_lymphoma_data_WSI_embeddings()
    classifier = ClassificationHead().to(device)
    classifier.load_state_dict(torch.load("/mnt/experiments/hipt_4k_extra_data/classifier.pt"))

    with torch.no_grad():
        correct = 0
        count_plots = 0
        per_class_count = {i: 0 for i in range(1, 7)}
        per_class_correct = {i: 0 for i in range(1, 7)}
        single_patch_per_wsi_counter = 0
        for i, data in enumerate(data_loader):
            X, y, name = data
            X = X.to(device).squeeze(dim=0)
            y = y.to(device).squeeze(dim=0)
            print("X shape: ", X.shape)
            if X.shape[0] == 10:
                print(f"Skipping WSI {name} with {X.shape[0]} patches")
                continue
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
            # if majority_vote != y.item() and y.item() == 4 and count_plots < 2 and X.shape[0] < 30 and X.shape[0] > 10:
            #     count_plots += 1
            #     # plot_wrong_patches(X, name, pred, y)
            #     plot_overlay(name, pred, y.item())
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
