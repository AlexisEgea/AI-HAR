import os
import json

import h5py


def load_data(path, filename):
    """ Load data with h5 extension """
    file = h5py.File(os.path.join(path, filename), 'r')
    return file.get('my_data')


def load_mocap_dataset_with_h5():
    """ Load dataset -> Features (x) + Labels (y) """
    config_path = os.path.join(os.getcwd(), "..", "data/dataset")
    x_filename = "train_mocap.h5"
    y_filename = "train_labels.h5"

    x = load_data(config_path, x_filename)
    y = load_data(config_path, y_filename)

    return x, y


def get_hyper_parameters_data():
    """Loads hyperparameters data from the configuration json file."""
    parameters_path = os.path.join(os.getcwd(), 'config/config.json')
    with open(parameters_path, 'r') as file:
        data = json.load(file)

    return data


def get_hyper_parameters(data, model_config, name):
    return data[model_config][name]
