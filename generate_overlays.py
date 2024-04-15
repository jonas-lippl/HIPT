import logging
import random
import sqlite3
from io import BytesIO

import numpy as np
import os

import timm
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F


INT2STR_LABEL_MAP = {
    0: "non-tumor",
    1: "tumor"
}


class UNI_classifier(torch.nn.Module):
    def __init__(self, n_classes=20):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.batch_norm = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, n_classes)
        # self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def choose_one_slide_per_class_for_testing() -> dict:
    """ Chooses one slide for each class for testing.

    Returns:
        dict: Dict containing the slide name and the corresponding label, one for each class.
    """
    slides = {"FL": None, "CLL": None, "HL": None, "DLBCL": None, "MCL": None, "Lts": None}
    all_slides = os.listdir("/data/sqlite")
    random.shuffle(all_slides)
    for slide in all_slides:
        if "HE" not in slide:
            continue
        diagnosis = slide.split(".")[0].split("-")[-1]
        if slides[diagnosis] is None:
            slides[diagnosis] = os.path.join("/data/sqlite", slide)
        else:
            continue
        if all([slide is not None for slide in slides.values()]):
            break
    return slides


#####################################
# FUNCTION TO GENERATE OVERLAY PLOT #
#####################################


def get_max_level(sqlite_path: str) -> int:
    """
    Gives the maximum level of tiles in the given sqlite file

    Args:
        sqlite_path (str): Path to the sqlite file

    Retruns:
        level (int): The maximal level of tiles in the sqlite file
    """
    con = sqlite3.connect(sqlite_path)
    cursor = con.cursor()

    cursor.execute('SELECT MAX(level) FROM tiles')
    level = cursor.fetchone()[0]
    con.close()

    return level


def get_coordinates_from_database(path: str, max_level: int) -> list | None:
    """
    Retrieve all coordinates from the 'tiles' table in a SQLite database.

        Args:
            path (str): The path to the SQLite database.
            max_level (int): The maximal zoom level of the slide.

        Returns:
            Tuple: A tuple containing all coordinates and the max level.
    """
    # get the coordinates from the tiles table if possible
    try:
        with sqlite3.connect(path) as conn:
            cursor = conn.cursor()
            # take all coordinates from max zoom level from the database
            cursor.execute("SELECT x, y FROM tiles WHERE level=?", (max_level,))
            coords = cursor.fetchall()
            cursor.close()
    except:
        logging.error("Error occurred while accessing tiles table for coordinates")
        raise Exception("Error occurred while accessing tiles table for coordinates")

    # check if there are any coordinates in the database
    if not coords:
        logging.warning("Set of coordinates found in tiles table is empty")
        return None
    return coords


def get_tile_from_database(cursor: sqlite3.Cursor, x: int, y: int, level: int) -> np.ndarray:
    """
    Retrieve one tile from the database for one set of coordinates.

    Args:
        cursor (sqlite3.Cursor): Cursor to execute SQL queries.
        x (int): current x-coordinate.
        y (int): current y-coordinate.
        level (int): Zoom level -> tiles are sampled from this level.

    Returns:
        np.ndarray: The tile as numpy array.
    """
    cursor.execute("SELECT jpeg FROM tiles WHERE x=? AND y=? AND level=? LIMIT 1", (x, y, level))
    tiles = cursor.fetchall()
    if len(tiles) == 1:
        # Create a BytesIO object from the image data
        bytes_io = BytesIO(tiles[0][0])
        # Read the image from the BytesIO object using PIL.Image.open()
        with Image.open(bytes_io) as pil_image:  # returns an image array in NumPy format
            img = np.array(pil_image)
    else:
        raise Exception(f"Error: Found more than one tile for the coordinates {x}, {y}.")
    return img


def generate_overlay_plot(
        slide_and_label: tuple,
        feature_extractor: torch.nn.Module,
        classifier: torch.nn.Module,
        device: torch.device,
        path_to_data: str,
        dpi: int
) -> None:
    """ Generates an overlay plot for one slide. The plot shows the patches of the slide and colors them according to the prediction of the model. The correct diagnosis is colored green.

    Args:
        slide_and_label (tuple): Tuple containing a label and an example slide for this label.
        feature_extractor (torch.nn.Module): Model for evaluation.
        classifier (torch.nn.Module): Model for evaluation.
        device (torch.device): Device on which the model is evaluated.
        path_to_data (str): Path to the data directory.
        dpi (int): DPI for saving the plot.
    """
    with torch.no_grad():
        slide_path = os.path.join(path_to_data, slide_and_label[1])
        correct_diagnosis = slide_and_label[0]
        level = get_max_level(sqlite_path=slide_path)
        print(f"Max level: {level}")
        coordinates = get_coordinates_from_database(slide_path, level)
        x_min, x_max, y_min, y_max = get_limits_of_plot(coordinates)
        print(f"Coordinate limits: {x_min}, {x_max}, {y_min}, {y_max}")

        # create empty plot on which we place the patches where they belong using the coordinates
        fig, ax = plt.subplots(1, 1, figsize=(30, 30))
        patch_size = 512
        ax.set_xlim(x_min*patch_size, x_max*patch_size)
        ax.set_ylim(y_min*patch_size, y_max*patch_size)
        ax.set_aspect('equal', adjustable='box')
        random.shuffle(coordinates)
        plot_patches(feature_extractor, classifier, device, coordinates, patch_size, correct_diagnosis,
                                  ax, level, slide_path)

        # add_labels_and_save_plot(ax, colors, correct_diagnosis, slide_and_label[1], dpi=dpi)


##############################################
# HELPER FUNCTIONS FOR GENERATE_OVERLAY_PLOT #
##############################################


def get_limits_of_plot(coordinates: list) -> tuple:
    """ Calculates the limits of the plot. """
    # get x and y coordinates of all patches
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    # get min and max values
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    return x_min, x_max, y_min, y_max


def plot_patches(feature_extractor: torch.nn.Module, classifier: torch.nn.Module, device: torch.device,
                 coordinates: list, patch_size: int, correct_diagnosis: int, ax: plt.Axes,
                 level: int, slide_path: str) -> None:
    # create list of colors for the different classes
    colors = ['green', 'blue']
    transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    count = 0
    name = slide_path.split('.')[0].split('/')[-1]
    save_path = f"./plots/{name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    with sqlite3.connect(slide_path) as conn:
        cursor = conn.cursor()
        # iterate over all patches and plot them on the empty plot
        for (x, y) in tqdm(coordinates, f"Patches for class {correct_diagnosis}"):
            img = get_tile_from_database(cursor, x, y, level)
            # print("Image shape: ", img.shape)
            tile = np.transpose(img, (2, 1, 0))
            tile = torch.from_numpy(tile)
            tile = transform(tile.div(255)).unsqueeze(0)
            # img = img / 255.0
            # ax.imshow(img, extent=(x*patch_size, x*patch_size + patch_size, y*patch_size, y*patch_size - patch_size))

            # get prediction of current patch
            tile = tile.to(device)
            embedding = feature_extractor(tile)
            output = classifier(embedding)
            _, pred = torch.max(output, 1)  # get the index of the class with the highest probability
            pred = pred.cpu().numpy()
            if pred[0] == 1:
                count += 1
                plt.imsave(f"{save_path}/tumor_pred_{count}.png", img)
            if count == 20:
                break
            # add colored square on top of the patch according to the prediction
            # rectangle = patches.Rectangle((x*patch_size, y*patch_size - patch_size), patch_size, patch_size, edgecolor='none',
            #                               facecolor=colors[pred[0]], alpha=0.1)
            # ax.add_patch(rectangle)
        cursor.close()

    # return ax, colors


def add_labels_and_save_plot(ax: plt.Axes, colors: list[str], correct_diagnosis: str,
                             slide_name: str, dpi: int) -> None:
    # add labels for all different predictions
    for i, color in enumerate(colors):
        ax.scatter([], [], marker=".", color=color, label=f"{INT2STR_LABEL_MAP[i]}")
    ax.legend(loc="lower right")
    ax.set_title(f"Correct diagnosis: {correct_diagnosis}")

    # save fig as vector graphic
    name = slide_name.split('.')[0].split('/')[-1] + "_overlay.png"
    path = os.path.join("./plots", name)
    print(f"Saving plot to {path}")
    plt.savefig(path, dpi=dpi)
    plt.close()


#################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate overlay plots for testing slides.")
    parser.add_argument("--dpi", type=int, default=500, help="DPI for saving the plots. (default: 500))")
    parser.add_argument("--path_to_data", type=str, default="/data",
                        help="Path to the data directory. (default: /data))")
    args = parser.parse_args()
    return dict(vars(args))


def main(dpi: int, path_to_data: str):
    # get list of all slides meant for testing and choose one slide for each class for testing
    test_slides_with_labels = choose_one_slide_per_class_for_testing()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model for evaluation
    torch.hub.set_dir('tmp/')
    local_dir = "./tmp/vit_large_patch16_256.dinov2.uni_mass100k"
    feature_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    feature_model.load_state_dict(torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu"), strict=True)
    feature_model.to(device)

    classifier = UNI_classifier(n_classes=2).to(device)
    classifier.load_state_dict(torch.load("./experiments/tissue_classifier_lr_decay/classifier.pt"))

    # iterate over different classes and generate overlay plots
    for slide_and_label in test_slides_with_labels.items():
        generate_overlay_plot(slide_and_label, feature_model, classifier, device, path_to_data, dpi=dpi)


"""
screen -dmS generate_overlay_plots sh -c 'docker run --shm-size=400gb --gpus \"device=1\"  -it --rm -u `id -u $USER` --name jol_job2 -v /sybig/home/jol/Code/HIPT:/mnt -v /sybig/projects/FedL/data:/data jol_hipt python3 /mnt/generate_overlays.py --dpi=1000 --path_to_data=./data/sqlite; exec bash'
"""

if __name__ == "__main__":
    args = parse_arguments()
    main(**args)
