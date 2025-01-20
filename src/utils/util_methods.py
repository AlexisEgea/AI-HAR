import os
import re

import numpy as np
import torch


def get_number_filename_to_save(base_path, filename):
    """ Return the next available number for a given filename in its directory. """
    files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
    numbers = []
    for file in files:
        match = re.search(rf"{filename}_(\d+)", file)
        if match:
            numbers.append(int(match.group(1)))

    if len(numbers) == 0:
        return 0
    return max(numbers) + 1


def get_classes(y_data):
    """ Retrieve the number of different labels (= classes number) from labels dataset."""
    return len(np.unique(y_data).tolist())


def add_weight(y):
    """ Return inverse class weights to address class imbalance during loss computation."""
    # Frequency of each class
    class_counts = torch.bincount(y[:, 0], minlength=get_classes(y))
    # Normalize the counts to get class weights (relative frequencies)
    class_weights = class_counts / len(y)

    print("Class Weights:", class_weights)

    # For balancing loss functions
    inverse_weights = 1.0 / class_weights
    print("Inverse Weights:", inverse_weights)

    return inverse_weights
