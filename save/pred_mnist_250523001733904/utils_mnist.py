import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle


def load_datasets(filename="dataset.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["datasets"], data["meta_variables"]


def build_domains(train_idx, test_idx, datasets, meta_vars, device):
    domains = []
    test_domains = []
    for idx in train_idx:
        X_i, Y_i = datasets[idx]
        meta = meta_vars[idx]
        domains.append((torch.tensor([meta], dtype=torch.float32, device=device),
                        torch.tensor(X_i, dtype=torch.float32, device=device).transpose(1, 3),
                        torch.Tensor(Y_i).long().to(device)))

    for idx in test_idx:
        X_i, Y_i = datasets[idx]
        meta = meta_vars[idx]
        test_domains.append((torch.tensor([meta], dtype=torch.float32, device=device),
                             torch.tensor(X_i, dtype=torch.float32, device=device).transpose(1, 3),
                             torch.Tensor(Y_i).long().to(device)))

    return domains, test_domains


def build_mlp_layers(input_dim=2, hidden_dim=50, output_dim=1):
    param_shapes = {
        "w1": (hidden_dim, input_dim),
        "b1": (hidden_dim,),
        "w2": (output_dim, hidden_dim),
        "b2": (output_dim,),
    }

    param_size = hidden_dim * input_dim + hidden_dim + output_dim * hidden_dim + output_dim
    return param_size, param_shapes


param_size, param_shapes = build_mlp_layers(576, 128, 10)


def forward_with_params(X, flat_params, param_shapes):
    offset = 0
    w1_size = param_shapes["w1"][0] * param_shapes["w1"][1]
    b1_size = param_shapes["b1"][0]
    w2_size = param_shapes["w2"][0] * param_shapes["w2"][1]
    b2_size = param_shapes["b2"][0]

    w1 = flat_params[offset: offset + w1_size]
    offset += w1_size
    b1 = flat_params[offset: offset + b1_size]
    offset += b1_size
    w2 = flat_params[offset: offset + w2_size]
    offset += w2_size
    b2 = flat_params[offset: offset + b2_size]
    offset += b2_size

    w1 = w1.view(param_shapes["w1"])
    b1 = b1.view(param_shapes["b1"])
    w2 = w2.view(param_shapes["w2"])
    b2 = b2.view(param_shapes["b2"])

    hidden = F.linear(X, w1, b1)
    hidden = torch.relu(hidden)
    hidden = F.linear(hidden, w2, b2)
    out = torch.log_softmax(hidden, dim=1)
    return out


predictive_model = nn.Sequential(
    nn.Linear(576, 128),
    nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1))

backbone_model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Dropout(0.7),
)


class Encoder(nn.Module):
    def __init__(self, param_dim, embed_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim))

    def forward(self, flat_params):
        return self.net(flat_params)


class Decoder(nn.Module):
    def __init__(self, embed_dim=8, param_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, param_dim))

    def forward(self, E):
        return self.net(E)
