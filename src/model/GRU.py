import sys
import os

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from model.base_model import BaseModel


class GRU(BaseModel):
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__(*args, **kwargs)

    def init_model(self):
        # GRU
        self.gru_layer = nn.GRU(input_size=self.n_input, hidden_size=self.n_hidden, num_layers=4, batch_first=True)

        self.output_layer = nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, x):
        # GRU
        gru_out, hidden_state = self.gru_layer(x)
        gru_out = torch.relu(gru_out)
        gru_output = gru_out[:, -1, :]
        out = self.output_layer(gru_output)

        return out

    def save_model(self):
        self.save_model_with_name("gru")