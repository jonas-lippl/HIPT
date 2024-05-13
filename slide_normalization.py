from typing import Union

import torch
from torchvision import transforms


class SlideNormalize:
    def __init__(self, slide_stats):
        self.slide_stats = slide_stats

    def __call__(self, sample):
        # Get the tile's slide
        image, label, slide_number = sample
        # Grab slide's mean and std
        mean, std = self.slide_stats[slide_number]
        # Normalize the tensor with slide specific values and convert to float
        normalized_image = transforms.Normalize(mean=mean, std=std)(image.float() / 255.)
        return normalized_image, label, slide_number


def normalize_dict(all_data: list[Union[torch.tensor, int, int]]) -> dict[int, Union[torch.tensor, torch.tensor]]:
    """
    Generates a dictionary with mean and std information for each slide (key)

    Args:
        all_data (list[Union[torch.tensor, int, int]]): blob, list of tuples (tensor, label, slide_label)

    Returns:
        dict[int, Union[torch.tensor, torch.tensor]]: keys are slide numbers, values are pairs (mean, std) already as toch.tensors
    """
    # Get the unique slide numbers
    slide_numbers = set(tensor[2] for tensor in all_data)

    # Initialize a dictionary to store the mean and std for each slide
    slide_stats = {}

    # For each slide number
    for iteration, slide_number in enumerate(slide_numbers):
        print(f"Training: Normalizing {((iteration) * 100) / len(slide_numbers):.1f}%".ljust(70), end="\r")
        # Get the indices of the tensors with this slide number
        indices = [i for i, tensor in enumerate(all_data) if tensor[2] == slide_number]

        # Get all tensors with this slide number
        tensors = [all_data[i][0].float() / 255. for i in indices]

        # Calculate mean and std of these tensors
        mean = torch.mean(torch.stack(tensors))
        std = torch.std(torch.stack(tensors))

        # Store the mean and std in the dictionary
        slide_stats[slide_number] = (mean.item(), std.item())

        #print(f"Slide {slide_number}, Mean = {mean:.4f}, Std = {std:.4f}")

        del tensors

    print(f"Training: Normalizing 100.0%".ljust(70), end="\r")

    return slide_stats
