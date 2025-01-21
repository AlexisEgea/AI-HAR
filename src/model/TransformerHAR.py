import sys
import os

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from model.base_model import BaseModel

class TransformerHAR(BaseModel):
    def __init__(self, *args, **kwargs):
        super(TransformerHAR, self).__init__(*args, **kwargs)

    def init_model(self):
        self.input_layer = nn.Linear(self.n_input, self.n_hidden)

        # Positional encoding for sequence information
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.n_steps, self.n_hidden))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_hidden, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.output_layer = nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, x):
        x = self.input_layer(x)

        x = x + self.positional_encoding[:, :x.size(1), :]

        x = self.transformer_encoder(x)

        x = torch.mean(x, dim=1)

        out = self.output_layer(x)

        return out

    def save_model(self):
        self.save_model_with_name("transformer_har")