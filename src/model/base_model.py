import os
import json
import sys
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.util_methods import get_number_filename_to_save

class BaseModel(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

        # Input Data
        self.n_steps = 0  # n timesteps per series
        self.n_input = 0  # n input parameters per timestep

        # Model internal structure
        self.n_hidden = 0  # Number of hidden features
        self.num_layers = 0  # Layers for LSTM/GRU/and so on
        self.n_classes = 0  # Total number of classes

        # Model saving information
        self.saved_model_name = self.retrieve_saved_model_name()
        self.save_model_path = os.path.join(os.getcwd(), "..", "data/saved_model")

    @abstractmethod
    def init_model(self):
        """ Initializes the concrete model """
        pass

    @abstractmethod
    def forward(self, x):
        """ Defines the forward pass of the concrete model """
        pass

    @abstractmethod
    def save_model(self):
        """ Saves the model """
        pass

    def get_classes(self):
        """ Returns the number of classes/outputs """
        return self.n_classes

    def init_hyper_parameters(self, n_steps, n_input, n_hidden, n_classes):
        """ Loads hyperparameters """
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def retrieve_saved_model_name(self):
        """ Retrieves the name of the saved model from the configuration file """
        parameters_path = os.path.join(os.getcwd(), 'config/config.json')
        with open(parameters_path, 'r') as file:
            data = json.load(file)

        return data['saved_model']['name']

    def save_model_with_name(self, name):
        """ Saves the model with a specific name """
        saved_model_name = f"{name}_{get_number_filename_to_save(self.save_model_path, name)}"

        path_model = os.path.join(self.save_model_path, saved_model_name)
        torch.save(self.state_dict(), path_model)

        print(f"model {saved_model_name} saved")

    def load_model(self, device):
        """ Loads the model from a saved file """
        path_model = os.path.join(self.save_model_path, self.saved_model_name)
        checkpoint = torch.load(path_model, map_location=torch.device(device))
        self.load_state_dict(checkpoint, strict=False)

        print(f"model {self.saved_model_name} loaded")
