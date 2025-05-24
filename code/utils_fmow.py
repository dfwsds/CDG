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


def build_domains(datasets, meta_vars, device):
    train_idx = np.array([1, 4, 5, 8, 10, 11, 13, 14, 19, 22, 23, 25, 26, 27, 28, 40, 41, 47, 48, 49, 53, 56, 60, 61, 62, 63, 69, 71, 74, 75, 76, 79, 81, 84, 88, 89, 90, 92, 93, 95])
    all_idx = np.arange(len(datasets))
    test_idx = np.setdiff1d(all_idx, train_idx)

    domains = []
    test_domains = []
    for idx in train_idx:
        X_i, Y_i = datasets[idx]
        meta = meta_vars[idx]
        domains.append((torch.tensor([meta], dtype=torch.float32, device=device),
                        torch.tensor(X_i, dtype=torch.float32, device=device),
                        torch.Tensor(Y_i).long().to(device)))

    for idx in test_idx:
        X_i, Y_i = datasets[idx]
        meta = meta_vars[idx]
        test_domains.append((torch.tensor([meta], dtype=torch.float32),
                             torch.tensor(X_i, dtype=torch.float32),
                             torch.Tensor(Y_i).long()))
    meta_vars_train = meta_vars[train_idx]
    return domains, test_domains, meta_vars_train


def build_mlp_layers(input_dim=2, hidden_dim=50, output_dim=1):
    param_shapes = {
        "w1": (hidden_dim, input_dim),
        "b1": (hidden_dim,),
        "w2": (32, hidden_dim),
        "b2": (32,),
        "w3": (output_dim, 32),
        "b3": (output_dim,),
    }

    param_size = hidden_dim * input_dim + hidden_dim + hidden_dim * 32 + 32 + output_dim * 32 + output_dim
    return param_size, param_shapes


param_size, param_shapes = build_mlp_layers(576, 128, 10)


def forward_with_params(X, flat_params, param_shapes):
    offset = 0
    w1_size = param_shapes["w1"][0] * param_shapes["w1"][1]
    b1_size = param_shapes["b1"][0]
    w2_size = param_shapes["w2"][0] * param_shapes["w2"][1]
    b2_size = param_shapes["b2"][0]
    w3_size = param_shapes["w3"][0] * param_shapes["w3"][1]
    b3_size = param_shapes["b3"][0]

    w1 = flat_params[offset: offset + w1_size]
    offset += w1_size
    b1 = flat_params[offset: offset + b1_size]
    offset += b1_size
    w2 = flat_params[offset: offset + w2_size]
    offset += w2_size
    b2 = flat_params[offset: offset + b2_size]
    offset += b2_size
    w3 = flat_params[offset: offset + w3_size]
    offset += w3_size
    b3 = flat_params[offset: offset + b3_size]
    offset += b3_size

    # 2) Reshape
    w1 = w1.view(param_shapes["w1"])
    b1 = b1.view(param_shapes["b1"])
    w2 = w2.view(param_shapes["w2"])
    b2 = b2.view(param_shapes["b2"])
    w3 = w3.view(param_shapes["w3"])
    b3 = b3.view(param_shapes["b3"])

    # 3) Forward
    hidden = F.linear(X, w1, b1)
    hidden = torch.relu(hidden)
    hidden = F.linear(hidden, w2, b2)
    hidden = torch.relu(hidden)
    hidden = F.linear(hidden, w3, b3)
    out = torch.log_softmax(hidden, dim=1)
    return out


predictive_model = nn.Sequential(
    nn.Linear(576, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(32, 10),
    nn.LogSoftmax(dim=1))

backbone_model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
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
