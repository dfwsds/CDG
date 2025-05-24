import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


def load_datasets(filename="dataset.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["datasets"], data["meta_variables"]


def build_domains(datasets, meta_vars, device):
    train_idx = np.array([0, 4, 11, 25, 29, 31, 33, 48, 49, 50, 51, 63, 65, 66, 71, 81, 82, 84, 91, 92, 94, 99, 103, 109, 113, 118, 120, 121, 122, 128, 129, 135, 142, 146, 152, 163, 183, 186, 194, 198, 202, 208, 209, 211, 214, 218, 226, 235, 242, 246, 249, 251, 252, 253, 256, 261, 267, 273, 280, 281, 284, 295, 297, 299, 301, 302, 309, 311, 318, 324, 327, 334, 342, 357, 360, 362, 368, 386, 396, 408, 432, 434, 435, 441, 446, 447, 448, 450, 451, 457, 458, 459, 464, 470, 471, 486, 491, 493, 498, 499, 506, 518, 520, 525, 528, 529, 531, 540, 546, 550, 556, 558, 569, 581, 588, 591, 593, 594, 601, 615, 616, 627, 628, 631, 635, 637, 639, 640, 651, 652, 653, 655, 666, 670, 682, 684, 685, 686, 687, 711, 712, 715, 748, 758, 759, 763, 772, 773, 774, 790, 793, 794, 795, 796, 797, 802, 803, 813, 817, 821, 822, 823, 825, 830, 835, 850, 853, 855, 857, 859, 878, 882, 883, 890, 902, 912, 917, 918, 920, 922, 923, 928, 930, 933, 934, 935, 936, 938, 940, 941, 949, 954, 956, 962, 971, 974, 978, 1008, 1019, 1023])
    all_idx = np.arange(len(datasets))
    test_idx = np.setdiff1d(all_idx, train_idx)

    domains = []
    test_domains = []
    for idx in train_idx:
        X_i, Y_i = datasets[idx]
        meta = meta_vars[idx]
        domains.append((torch.tensor([meta], dtype=torch.float32, device=device),
                        torch.tensor(X_i, dtype=torch.float32, device=device).view(X_i.shape[0], -1),
                        torch.tensor(Y_i, device=device).view(Y_i.shape[0], -1)))

    for idx in test_idx:
        X_i, Y_i = datasets[idx]
        meta = meta_vars[idx]
        test_domains.append((torch.tensor([meta], dtype=torch.float32, device=device),
                             torch.tensor(X_i, dtype=torch.float32, device=device).view(X_i.shape[0], -1),
                             torch.tensor(Y_i, device=device).view(Y_i.shape[0], -1)))
    meta_vars_train = meta_vars[train_idx]
    return domains, test_domains, meta_vars_train


def build_mlp_layers(input_dim=48 * 2, hidden_dim=64, output_dim=3 * 2):
    """
    DLinear per-domain predictor.
    """
    param_shapes = {
        "w1": (hidden_dim, input_dim),
        "b1": (hidden_dim,),
        "w2": (hidden_dim, hidden_dim),
        "b2": (hidden_dim,),
        "w3": (output_dim, hidden_dim),
        "b3": (output_dim,),
    }

    param_size = (
            hidden_dim * input_dim +
            hidden_dim +
            hidden_dim * hidden_dim +
            hidden_dim +
            output_dim * hidden_dim +
            output_dim
    )
    return param_size, param_shapes


param_size, param_shapes = build_mlp_layers()


def forward_with_params(X, flat_params, param_shapes):
    offset = 0
    w1_size = np.prod(param_shapes["w1"])
    b1_size = np.prod(param_shapes["b1"])
    w2_size = np.prod(param_shapes["w2"])
    b2_size = np.prod(param_shapes["b2"])
    w3_size = np.prod(param_shapes["w3"])
    b3_size = np.prod(param_shapes["b3"])

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

    w1 = w1.view(param_shapes["w1"])
    b1 = b1.view(param_shapes["b1"])
    w2 = w2.view(param_shapes["w2"])
    b2 = b2.view(param_shapes["b2"])
    w3 = w3.view(param_shapes["w3"])
    b3 = b3.view(param_shapes["b3"])

    X = X.view(X.size(0), -1)
    hidden = F.linear(X, w1, b1)
    hidden = F.relu(hidden)
    hidden = F.linear(hidden, w2, b2)
    hidden = F.relu(hidden)
    out = F.linear(hidden, w3, b3)
    return out


predictive_model = nn.Sequential(
    nn.Linear(48 * 2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 3 * 2), )

backbone_model = nn.Sequential()


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
