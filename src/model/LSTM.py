import sys
import os

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from model.base_model import BaseModel


class LSTM(BaseModel):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)

    def init_model(self):
        # LSTM
        self.lstm_layer = nn.LSTM(input_size=self.n_input, hidden_size=self.n_hidden, num_layers=4, batch_first=True)

        self.output_layer = nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, x):
        lstm_out, (hidden_state1, cell_state1) = self.lstm_layer(x)
        lstm_out = torch.relu(lstm_out)
        # many-to-one
        lstm_output = lstm_out[:, -1, :]
        out = self.output_layer(lstm_output)

        return out

    def save_model(self):
        self.save_model_with_name("lstm")
