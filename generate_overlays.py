import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import IterableDataset, TensorDataset 
from itertools import chain
from pamly import Diagnosis, Stain

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys
import utils

INT2STR_LABEL_MAP = {
    int(Diagnosis("Unknown")): "Unknown",
    int(Diagnosis("HL")): "HL",
    int(Diagnosis("DLBCL")): "DLBCL",
    int(Diagnosis("CLL")): "CLL",
    int(Diagnosis("FL")): "FL",
    int(Diagnosis("MCL")): "MCL",
    int(Diagnosis("LTDS")): "Lts",
}


def retrieve_list_of_all_test_slides(path_to_splits: str) -> list:
    """ Retrieves the list of all slides meant for testing from the splits csv file. """
    # load splits csv file from output directory
    if os.path.exists(path_to_splits):
        splits_df = pd.read_csv(path_to_splits)
    else:
        raise FileNotFoundError(f"File {path_to_splits} does not exist.")
    # get all slide names meant for testing from splits.csv, i.e. all rows where the column "test" is 1 and get rid of the '.sqlite' ending 
    all_test_slides = splits_df["filenames"][splits_df["test"] == 1].values # .values returns a numpy array instead of a pandas series
    all_test_slides = [slide[:-7] for slide in all_test_slides]
    return all_test_slides, splits_df



def choose_one_slide_per_class_for_testing(splits_df: pd.DataFrame, all_test_slides: list, path_to_data: str, minimal_nr_of_patches: int = 1000, maxtries=10e14) -> list:
    """ Chooses one slide for each class for testing.

    Args:
        splits_df (pd.DataFrame): Dataframe containing the splits csv file.
        all_test_slides (list): List of all slides meant for testing extracted from splits csv file.
        path_to_data (str): Path to the data directory.

    Returns:
        list: List of lists containing the slide name and the corresponding label, one for each class.
    """
    nr_of_classes = splits_df["diagnosis"].nunique()
    test_slides_with_labels = []
    diagnosis = []
    index=0
    while len(test_slides_with_labels) < nr_of_classes:
        if index > maxtries:
            raise RuntimeError(f"Could not find slides with at least {minimal_nr_of_patches} patches for testing after {maxtries} tries.")
        # choose slide from all_test_slides
        index += 1
        slide = all_test_slides[index]
        # get diagnosis of current slide
        df = pd.read_csv(os.path.join(path_to_data, slide + ".csv"))
        label = df["diagnosis"][0]
        # check if diagnosis is already in test_slides_with_labels
        if label not in diagnosis and len(df) > minimal_nr_of_patches:
            test_slides_with_labels.append([slide, label])
            diagnosis.append(label)
    # sort test_slides_with_labels according to the labels
    test_slides_with_labels.sort(key=lambda x: x[1])
    print(f"Chosen slides for testing: {test_slides_with_labels}")
    return test_slides_with_labels


###########################################################################################################################
######################################## FUNCTION TO GENERATE OVERLAY PLOT ################################################
###########################################################################################################################


def generate_overlay_plot(
        slide_and_label: list, 
        model: torch.nn.Module, 
        device: torch.device, 
        path_to_data: str, 
        output_dir: str, 
        batch_size: int, 
        plots_dir: str,
        dpi: int
        ) -> None:
    """ Generates an overlay plot for one slide. The plot shows the patches of the slide and colors them according to the prediction of the model. The correct diagnosis is colored green.

    Args:
        slide_and_label (list): List containing the slide name and the corresponding label.
        model (torch.nn.Module): Model for evaluation.
        device (torch.device): Device on which the model is evaluated.
        path_to_data (str): Path to the data directory.
        output_dir (str): Path to the output directory.
        batch_size (int): Batch size for evaluation.
        plots_dir (str): Name of the directory in which the plots are saved.
    """
    with torch.no_grad():
        # setup needed for iterable dataset 
        test_len = get_number_of_testing_samples(slide_and_label, path_to_data)
        transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        # get diagnosis and coordinates of patches for plotting and calculate the limits of the plot
        file_path = os.path.join(path_to_data, slide_and_label[0] + ".csv")
        correct_diagnosis = pd.read_csv(file_path)["diagnosis"][0]
        coordinates = get_coordinates_from_csv(file_path, tile_size=512)
        x_min, x_max, y_min, y_max = get_limits_of_plot(coordinates)

        # create empty plot on which we place the patches where they belong using the coordinates
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')

        # get the original patch size in pixels from log file (before resizing happend)
        path_to_log = os.path.join(path_to_data, slide_and_label[0] + ".log")
        patch_size = retrieve_patch_size_from_log(path_to_log)

        test_loader = get_test_data(slide_and_label, path_to_data, test_len, transform, batch_size)
        ax, colors = plot_patches(test_loader, model, device, coordinates, patch_size, correct_diagnosis, ax)

        add_labels_and_save_plot(ax, colors, correct_diagnosis, output_dir, plots_dir, slide_and_label[0], dpi=dpi)

###########################################################################################################################
###################################### HELPER FUNCTIONS FOR GENERATE_OVERLAY_PLOT #########################################
###########################################################################################################################

def get_number_of_testing_samples(slide_and_label: list, path_to_data: str) -> int:
    """ Get total number of samples for testing for one testing slides; needed for dataloader. """
    # get number of samples for current slide
    slide = slide_and_label[0]
    df = pd.read_csv(os.path.join(path_to_data, slide + ".csv"))
    test_len = len(df)
    print(f"Number of testing samples for {slide}: {test_len}")
    return test_len

def get_coordinates_from_csv(path_to_csv: str, tile_size: int = 512) -> list:
    """ Retrieves the coordinates of the patches of a slide needed for plotting from the slides csv file. The coordinates are transformed to absolute coordinates. 
    
    Args:
        path_to_csv (str): Path to the csv file containing the coordinates of the patches.
        tile_size (int, optional): Size of the tiles in pixels. Defaults to 512.

    Returns:
        list: List of tuples containing the coordinates of the patches.
    """
    df = pd.read_csv(path_to_csv)
    coordinates = df["coordinates"].values # form  ['((20, 40), [143, 488])' '((21, 35), [30, 413])' ...]
    # convert coordinates to list of tuples and lists of ints
    coordinates = [eval(coord) for coord in coordinates] # form [(20, 40), [143, 488]) (21, 35), [30, 413]) ...]
    # compute the absolute coordinates, i.e, combine tile coordinates and patch coordinates 
    coordinates = [(coord[0][0]*tile_size + coord[1][0], coord[0][1]*tile_size + coord[1][1]) for coord in coordinates]
    return coordinates

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

def retrieve_patch_size_from_log(path_to_log: str) -> int:
    """ Retrieves the patch size in pixels from the log file. """
    # get the original patch size in pixels from log file, before resizing happend
    with open(path_to_log, "r") as f:
        lines = f.readlines()
        # search for lines containing Patch size in micro meters and Pixels per m of the WSI
        for line in lines:
            if "Patch size in micro meter" in line:
                patch_size_um = int(line.split(":")[-1])
            if "Pixels per m of the WSI" in line:
                pixels_per_m = int(line.split(":")[-1])
    # calculate the original patch size in pixels
    patch_size = int(patch_size_um * pixels_per_m / 1e6)
    return patch_size

def get_test_data(slide_and_label: list, path_to_data: str, test_len: int, transform: transforms.Compose, batch_size: int) -> DataLoader:
    """ Get data loader for testing. """
    file_path = [os.path.join(path_to_data, slide_and_label[0] + ".pt")] # this form needed for MyIterableTestDataset
    test_data_iter = MyIterableTestDataset(path_to_data, file_path, test_len, transform=transform)
    test_loader = DataLoader(test_data_iter, batch_size=batch_size, drop_last=True, num_workers=1, prefetch_factor=4)
    return test_loader

def plot_patches(test_loader: DataLoader, model: torch.nn.Module, device: torch.device, coordinates: list, patch_size: int, correct_diagnosis: int, ax: plt.Axes) -> tuple[plt.Axes, list[str]]:
    # create list of colors for the different classes
    colors = ['green', 'blue', 'orange', 'purple', 'brown', 'yellow']

    # iterate over all patches and plot them on the empty plot
    for i, (x, _) in enumerate(test_loader):
        coord = coordinates[i]
        img = prepare_transformed_tensor_for_plotting(x)
        ax.imshow(img, extent=[coord[0], coord[0]+patch_size, coord[1], coord[1]-patch_size])

        # get prediction of current patch
        x = x.to(device)
        output = model(x).softmax(1) # apply softmax to get probabilities
        _, pred = torch.max(output, 1) # get the index of the class with the highest probability 
        pred = pred.cpu().numpy()
        
        # add colored square on top of the patch according to the prediction
        rectangle = patches.Rectangle((coord[0], coord[1]-patch_size), patch_size, patch_size, edgecolor='none', facecolor=colors[pred[0]], alpha=0.4)
        ax.add_patch(rectangle)

        print(f"Processed patches for class {correct_diagnosis}: {i+1}/{len(test_loader)}", end="\r")

    print()

    return ax, colors

def add_labels_and_save_plot(ax: plt.Axes, colors: list[str], correct_diagnosis: int, output_dir: str, plots_dir: str, slide_name: str, dpi: int) -> None:
    # add labels for all different predictions
    for i, color in enumerate(colors):
        ax.scatter([], [], marker=".", color=color, label=f"{INT2STR_LABEL_MAP[i]}")
    ax.legend(loc="lower right")
    ax.set_title(f"Correct diagnosis: {INT2STR_LABEL_MAP[correct_diagnosis]}")
    
    # save fig as vector graphic
    name = slide_name + "_overlay.png"
    path = os.path.join(output_dir, plots_dir, name)
    plt.savefig(path, dpi=dpi)
    plt.close()


################################ helper functions for plot_patches ###########################################

def prepare_transformed_tensor_for_plotting(x: torch.Tensor) -> torch.Tensor:
    """ Gets image of current patch and undoes the normalization with mean and std of the used pretrained model """
    # take first patch of batch with size 1
    img = x[0] 
    transform = transforms.Compose([transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])])
    img = transform(img)
    # permute to get the right format for plt.imshow
    img = img.permute(2, 1, 0) # such that it is in the right format for plt.imshow
    return img



###########################################################################################################################
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate overlay plots for testing slides.")
    parser.add_argument("--offset", type=int, default=0, help="Offset for distributed training. (default: 0)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation. (default: 1)")
    parser.add_argument("--dpi", type=int, default=500, help="DPI for saving the plots. (default: 500))")
    parser.add_argument("--path_to_data", type=str, default="/data", help="Path to the data directory. (default: /data))")
    parser.add_argument("--output_dir", type=str, default="./training_runs/03_run", help="Path to the output directory. (default: ./training_runs/03_run))")
    parser.add_argument("--path_to_model", type=str, default="best_model.pt", help="Path to the model. (default: best_model.pt))")
    parser.add_argument("--filename_splits", type=str, default="splits.csv", help="Filename of splits csv file. (default: splits.csv))")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Name of the directory in which the plots are saved. (default: plots)))")
    args = parser.parse_args()
    return dict(vars(args))


def main(offset: int, batch_size: int, dpi: int, path_to_data: str, output_dir: str, path_to_model: str, filename_splits: str, plots_dir: str):
    # setup 
    utils.general_setup()
    utils.ddp_setup(offset)
    
    print(f"Using device {device}")
    path_to_splits = os.path.join(output_dir, filename_splits)

    # rename blobs if necessary -> this is only needed once such that the blobs have the same name as the corresponding slide
    rename_blobs(blob_dir=path_to_data)

    # get list of all slides meant for testing and choose one slide for each class for testing
    all_test_slides, splits_df = retrieve_list_of_all_test_slides(path_to_splits)
    test_slides_with_labels = choose_one_slide_per_class_for_testing(splits_df, all_test_slides, path_to_data)

    # load model for evaluation
    path = os.path.join(output_dir, path_to_model)
    model = utils.load_trained_model(path, device)
    model = model.to(device) 
    model.eval()
    
    # iterate over different classes and generate overlay plots
    for slide_and_label in test_slides_with_labels:
        generate_overlay_plot(slide_and_label, model, device, path_to_data, output_dir, batch_size, plots_dir, dpi=dpi)



    

        
"""
docker run --shm-size=100gb --gpus all -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code/lymphoma/Resnet_ddp:/mnt -v /sybig/home/ftoelkes/preprocessed_data/all_slides_test_200um:/data fto_resnet torchrun --standalone --nproc_per_node=1 /mnt/eval_model/generate_overlays.py --offset=0 --batch_size=1 --dpi=1500 --path_to_data=/data --output_dir=./training_runs/200um_models/model_100k_patches_200_epochs --path_to_model=best_model_params.pt"""


if __name__=="__main__":
    args = parse_arguments()
    main(**args)
    

    