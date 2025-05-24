import argparse
import shutil
import json
import importlib
import os
import sys
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import compute_classification_metrics, compute_regression_metrics


# ---------------------------
def load_config(dataset_name, config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config[dataset_name]


# ---------------------------
def get_dataset_module(dataset_name):
    mapping = {
        "moons": "utils",
        "mnist": "utils_mnist",
        "arxiv": "utils_arxiv",
        "fmow": "utils_fmow",
        "yearbook": "utils_yearbook",
        "traffic": "utils_traffic",
    }
    return importlib.import_module(mapping[dataset_name])


def evaluate_test_domains(train_domains, test_domains, transport_net, decoder, param_shapes, encoder, theta_list,
                          shared_model, dataset_module, train_idx, neighbor_model):
    prediction, label = [], []
    shared_model.eval()

    for idx, (rs_tensor, X_star, Y_star) in enumerate(test_domains):
        rs_tensor, X_star, Y_star = rs_tensor.to(device), X_star.to(device), Y_star.to(device)
        with torch.no_grad():
            X_star = shared_model(X_star)

            meta_query = rs_tensor.cpu().numpy().reshape(1, -1)
            distances, indices = neighbor_model.kneighbors(meta_query)
            ref_idx = train_idx[indices[0][0]]

            rs0, e0 = train_domains[ref_idx][0], theta_list[ref_idx]

            E_star = transport_net(rs_tensor, rs0, encoder(e0).unsqueeze(0))
            theta_star = decoder(E_star).squeeze(0)
            pred = dataset_module.forward_with_params(X_star, theta_star, param_shapes)

            prediction.append(pred.detach().cpu().numpy())
            label.append(Y_star.cpu().numpy())
    return prediction, label


class NeuralLTO(nn.Module):
    def __init__(self, dim, meta_input_dim, n_mani=2, hidden=128):
        super().__init__()
        self.dim = dim
        self.n_mani = n_mani

        self.field_net = nn.Sequential(
            nn.Linear(meta_input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_mani * dim * dim),
        )

        self.meta_embed = nn.Sequential(
            nn.Linear(meta_input_dim, n_mani, bias=False),
        )

    def forward(self, xy, xy0, e0):
        L_all = self.field_net(xy0).view(self.n_mani, self.dim, self.dim)
        diff = xy - xy0

        e_trans = e0
        for i in range(diff.shape[1]):
            Li = L_all[i]
            exp_Li = torch.matrix_exp(diff[:, i] * Li)
            e_trans = torch.matmul(e_trans, exp_Li.t())
        return e_trans


# ---------------------------
def main(args, device):
    dataset_module = get_dataset_module(args.dataset)
    datasets, meta_vars = dataset_module.load_datasets(args.data_path)
    meta_vars = np.array(meta_vars)
    if args.dataset in ["moons", "mnist"]:
        meta_vars = meta_vars - meta_vars.min(axis=0)
        meta_vars = meta_vars / meta_vars.max(axis=0)

    # Build Domain Datasets
    train_idx = np.arange(50)
    all_idx = np.arange(len(datasets))
    test_idx = np.setdiff1d(all_idx, train_idx)

    domains, test_domains = dataset_module.build_domains(train_idx, test_idx, datasets, meta_vars, device)
    train_idx = [i for i in range(len(domains))]

    # Build Local Chart
    n_neighbors = 5
    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(meta_vars[train_idx])
    _, indices = neighbor_model.kneighbors(meta_vars[train_idx])
    neighbors = {idx: indices[i] for i, idx in enumerate(train_idx)}

    # Initialization
    num_domains = len(domains)
    meta_input_dim = meta_vars.shape[1]
    param_size = dataset_module.param_size
    param_shapes = dataset_module.param_shapes

    transport_net = NeuralLTO(32, meta_input_dim).to(device)
    shared_model = dataset_module.backbone_model.to(device)
    predictive_model = dataset_module.predictive_model
    domain_models = nn.ModuleList([predictive_model.to(device) for _ in range(num_domains)])
    theta_list = nn.ParameterList([nn.Parameter(torch.cat([p.flatten() for p in m.parameters()])) for m in domain_models])
    encoder = dataset_module.Encoder(param_dim=param_size, embed_dim=32).to(device)
    decoder = dataset_module.Decoder(embed_dim=32, param_dim=param_size).to(device)

    optimizer = torch.optim.Adam([
        {'params': shared_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': theta_list.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': encoder.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': decoder.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': transport_net.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}
    ])

    # Training
    lw1, lw2, lw3, lw4, lw5 = 1, 1, 100, 10, 100
    num_epochs = 200
    best_epoch_loss = np.inf

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        shared_model.train()
        for batch_idx in np.array_split(train_idx, len(train_idx) // 5):
            optimizer.zero_grad()

            loss_1, loss_2, loss_recon, loss_embed, loss_consis = 0.0, 0.0, 0.0, 0.0, 0.0

            for idx in batch_idx:
                rs_i, X_i, Y_i = domains[idx] # descriptor, X, Y
                X_i = shared_model(X_i)

                theta_i = theta_list[idx]
                e_i = encoder(theta_i.unsqueeze(0))
                theta_i_recon = decoder(e_i).squeeze(0)

                l_rec = F.mse_loss(theta_i_recon, theta_i)
                loss_recon += l_rec

                for neighbor_idx in neighbors[idx]:
                    rs_neighbor, X_neighbor, Y_neighbor = domains[neighbor_idx]
                    theta_neighbor = theta_list[neighbor_idx]
                    X_neighbor = shared_model(X_neighbor)
                    e_neighbor = encoder(theta_neighbor.unsqueeze(0))

                    e_neighbor_pred = transport_net(rs_neighbor, rs_i, e_i)
                    theta_neighbor_pred = decoder(e_neighbor_pred).squeeze(0)

                    if args.dataset in ["moons", "mnist", "fmow", "arxiv", "yearbook"]:
                        l_j = F.nll_loss(dataset_module.forward_with_params(X_neighbor, theta_neighbor_pred, param_shapes), Y_neighbor)
                        l_i = F.nll_loss(dataset_module.forward_with_params(X_i, theta_i, param_shapes), Y_i)
                    else:
                        l_j = F.l1_loss(dataset_module.forward_with_params(X_neighbor, theta_neighbor_pred, param_shapes), Y_neighbor)
                        l_i = F.l1_loss(dataset_module.forward_with_params(X_i, theta_i, param_shapes), Y_i)

                    loss_1 += l_i
                    loss_2 += l_j

                    l_embed = F.mse_loss(e_neighbor, e_neighbor_pred)
                    loss_embed += l_embed

                    l_consis = F.mse_loss(theta_neighbor, theta_neighbor_pred)
                    loss_consis += l_consis

            total_loss = lw1 * loss_1 + lw2 * loss_2 + lw3 * loss_recon + lw4 * loss_embed + lw5 * loss_consis
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss

        if (epoch + 1) % 1 == 0:
            log_str = f"Epoch {epoch + 1}/{num_epochs}, Total Loss={total_loss.item():.4f}, "
            log_str += f"Loss1={loss_1.item():.4f}, Loss2={loss_2.item():.4f}, Recon={loss_recon.item():.4f}, "
            log_str += f"Embed={loss_embed.item():.4f}, Consis={loss_consis.item():.4f}"
            print(log_str)
            with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                f.write(log_str + "\n")

        if ((epoch + 1) % 10 == 0) and (epoch_loss < best_epoch_loss):
            checkpoint = {
                "shared_model": shared_model.state_dict(),
                "theta_list": theta_list.state_dict(),
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "transport_net": transport_net.state_dict(),
            }
            torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
            print(f"Model saved to {args.save_dir}")
            best_epoch_loss = epoch_loss

    # Evaluation
    checkpoint = torch.load(args.save_dir + '/checkpoint.pth')
    transport_net.load_state_dict(checkpoint["transport_net"])
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    shared_model.load_state_dict(checkpoint["shared_model"])
    theta_list.load_state_dict(checkpoint["theta_list"])

    test_pred, test_label = evaluate_test_domains(domains, test_domains, transport_net, decoder, param_shapes, encoder, theta_list, shared_model, dataset_module, train_idx, neighbor_model)
    if args.dataset in ["moons", "mnist", "fmow", "arxiv", "yearbook"]:
        metrics = compute_classification_metrics(test_label, test_pred)
    else:
        metrics = compute_regression_metrics(test_label, test_pred)
    score_log = f"{metrics}"
    print(score_log)
    with open(os.path.join(args.save_dir, "score.txt"), "a") as f:
        f.write(score_log + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="GPU")
    parser.add_argument("--dataset", type=str, default="moons",
                        choices=["moons", "mnist", "fmow", "arxiv", "yearbook", "traffic"])

    args = parser.parse_args()
    args.data_path = './data/Moons/dataset.pkl'
    args.save_dir = './save/' + 'pred_moons_' + datetime.now().strftime("%y%m%d%H%M%S%f")[:-3]
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, args.save_dir)
    shutil.copy2('utils.py', args.save_dir)
    shutil.copy2('metrics.py', args.save_dir)

    main(args, device)
