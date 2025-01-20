import numpy as np
import torch

from utils.plotting import plot_3d_video, plot_data_3d_frame
from utils.util_methods import get_classes


def process_mocap_data(x, y, segment_size):
    """
    For Features (x):
        1. Remove Pelvis + Subtract Pelvis
        2. Normalize Data with the highest and lowest value of the dataset
        3. -> np.array
        4. -> Tensor(type=float32)

    For Labels (y):
        1. -> np.array
        2. -> Tensor(type=Long)
        3. Encode Labels with one hot method + type=float
        4. torch.Size([6938, 1, 13]) -> torch.Size([6938, 13])
    """
    x_data = preparing_mocap_dataset(x, segment_size)

    x_data = np.array(x_data)
    x_data_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_data = np.array(y)
    # Long for one hot encoding method
    y_data_tensor = torch.tensor(y_data, dtype=torch.long)
    y_data_tensor = torch.nn.functional.one_hot(y_data_tensor, num_classes=get_classes(y)).float()
    # torch.Size([6938, 1, 13]) -> torch.Size([6938, 13])
    y_data_tensor = y_data_tensor.squeeze(1)
    # x_data_tensor and y_data_tensor -> Float type
    return x_data_tensor, y_data_tensor


def preparing_mocap_dataset(x, segment_size):
    """  Prepare the motion capture (MoCap) dataset by segmenting/centering and normalizing ([-1,1]) data. """
    # (6938, 12900) -> (6938, 100, 129)
    x_data = []
    for window in x:
        x_data.append([window[i:i + segment_size] for i in range(0, len(window), segment_size)])

    print(f"x_data before preparation= {x_data[0][0][:3]}")
    plot_data_3d_frame(x_data[0][0])
    plot_3d_video(x_data[0])
    x_data = remove_pelvis(x_data)
    print(f"x_data after removing Pelvis= {x_data[0][0][:3]}")
    # Uncomment the next two lines to see an example of mocap without pelvis and new dimension by subtracting it
    # plot_data_3d_frame(x_data[0][0])
    # plot_3d_video(x_data[0])
    x_data = normalize_data(x_data)
    print(f"x_data after normalisation= {x_data[0][0][:3]}")
    # plot_data_3d_frame(x_data[0][0])
    plot_3d_video(x_data[0])

    return x_data


def remove_pelvis(x):
    """ Remove Pelvis from Features (x) + Subtract it to features"""
    x = np.array(x)
    # Extract pelvis coordinates (last 3 features)
    pelvis = x[:, :, -3:]  # Shape: (6938, 100, 3)
    # Remove pelvis from the data
    x = x[:, :, :-3]  # (6938, 100, 129) -> (6938, 100, 126)
    # Repeat pelvis coordinates across features
    pelvis_repeated = np.tile(pelvis, 42)  # Shape: (6938, 100, 3, 42)
    # Normalize data by subtracting pelvis
    x = x - pelvis_repeated

    # For debuging purposes
    # print(f"Pelvis= {pelvis[0, 0]}")

    return x.tolist()


def normalize_data(x):
    """ Normalize Data by computing the global min and max of the entire dataset """
    x = np.array(x)
    # Reshape into (total_entries, features), collapsing samples and timesteps
    x_flat = x.reshape(-1, x.shape[-1])  # (6938, 100, 126) -> (693800, 126)
    # Compute global Min and Max for features
    min_values = np.min(x_flat)
    max_values = np.max(x_flat)

    # TODO: Doesn't work for now
    # Compute global Min and Max for each feature
    # min_values = np.min(x_flat, axis=0)
    # max_values = np.max(x_flat, axis=0)

    # TODO : Not needed anymore
    # Avoid division by zero
    # diff = max_values - min_values
    # diff[diff == 0] += 1e-6  # Add small epsilon to prevent division by zero

    # Normalize data to [-1, 1]
    x_normalized = 2 * (x_flat - min_values) / (max_values - min_values) - 1

    # Reshape back to the original structure
    x_normalized = x_normalized.reshape(x.shape)

    return x_normalized.tolist()
