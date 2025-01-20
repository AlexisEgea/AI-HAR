import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

import matplotlib.pyplot as plt


def plot_learning_curve(losses, accuracies):
    """ Plots the progress of loss and accuracy over iterations """
    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()
    ax1.plot(losses, label='Loss', color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.plot(accuracies, label='Accuracy', color='blue')
    ax2.set_ylabel('Accuracy', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title('Loss and Accuracy Curve')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_training_progress(train_losses, train_accuracies, test_losses, test_accuracies):
    """ Plots the progress of training (loss and accuracy) over iterations """
    font = {
        'family': 'Bitstream Vera Sans',
        'weight': 'bold',
        'size': 18
    }
    plt.rc('font', **font)

    width = 12
    height = 12
    fig = plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(len(train_losses)))

    plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

    indep_test_axis = np.array(range(0, len(test_losses)))
    plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Validation losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Validation accuracies")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')

    plt.show()

    return fig


def plot_data_3d_frame(data, title="3D Data Plot for Frame", labels=None):
    """ Plots a single frame of 3D data (number of sensors × 3 coordinates) """
    # Convert data to np.array if it is a list or torch.Tensor
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.numpy()

    # Reshape the data into (number_of_sensors, 3) if in flat format
    if data.ndim == 1:
        data = data.reshape(-1, 3)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if labels is not None and len(labels) == data.shape[0]:
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="viridis", s=50)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Sensor Labels")
    else:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='b', s=50)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(elev=20, azim=30)
    plt.show()


def plot_3d_video(data, title="3D Video of Data", window=100):
    """ Plots a video of 3D data (number of sensors × 3 coordinates) """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.numpy()

    n_frames = data.shape[0]
    data = data.reshape(n_frames, -1, 3)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    x_limits = (np.min(data[:, :, 0]), np.max(data[:, :, 0]))
    y_limits = (np.min(data[:, :, 1]), np.max(data[:, :, 1]))
    z_limits = (np.min(data[:, :, 2]), np.max(data[:, :, 2]))

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Function for animation
    def update(frame):
        ax.clear()
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        ax.set_title(f"{title} - Frame {frame + 1}/{n_frames}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        scatter = ax.scatter(data[frame][:, 0], data[frame][:, 1], data[frame][:, 2], color='b', s=50)
        return scatter,

    animation = FuncAnimation(fig, update, frames=n_frames, interval=window, blit=False)
    plt.show()
