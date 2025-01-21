import sys
import os

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from model.base_model import BaseModel


class LSTM_RNN(BaseModel):
    def __init__(self, *args, **kwargs):
        super(LSTM_RNN, self).__init__(*args, **kwargs)

    def init_model(self):
        print(self.n_input)
        print(self.n_hidden)

        # Bidirectional LSTM Layers
        self.lstm = nn.LSTM(self.n_input, self.n_hidden, 2, bidirectional=True, batch_first=True)

        # Residual layers, ReLU, and BatchNorm
        self.residual_layers = nn.ModuleList([nn.Identity() for _ in range(2)])
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.n_hidden * 2)

        self.output_layer = nn.Linear(self.n_hidden * 2, self.n_classes)

    def forward(self, x):
        print(f"before= {x.shape}")
        output, _ = self.lstm(x)

        print(f"lstm= {output.shape}")

        for i in range(2):
            print(f"{i} output before residual= {output.shape}")
            residual = self.residual_layers[i](output)
            print(f"residual {i}= {output.shape}")
            output = self.relu(output + residual)
            print(f"relu {i}= {output.shape}")

            output = output.reshape(-1, self.n_hidden * 2)
            print(f"reshape {i}= {output.shape}")

            output = self.bn(output)
            print(f"norm {i}= {output.shape}")

            output = output.reshape(self.batch_size, -1, self.n_hidden * 2)
            print(f"reshape {i}= {output.shape}")

        output = output[:, -1, :]
        print(f"output= {output.shape}")
        output = self.output_layer(output)
        print(f"output= {output.shape}")

        return output

    def save_model(self):
        self.save_model_with_name("lstm_rnn")


