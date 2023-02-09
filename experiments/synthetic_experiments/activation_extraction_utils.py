import numpy as np

import torch.nn as nn
from torch_geometric.nn import GCNConv, DenseGCNConv


def get_activation(idx, feature_dict):
    def hook(model, input, output):
        feature_dict[idx] = output.detach()
    return hook


def register_hooks(model, feature_dict):
    for name, m in model.named_modules():
        if isinstance(m, GCNConv) or isinstance(m, DenseGCNConv):
            m.register_forward_hook(get_activation(f"{name}", feature_dict))
        if isinstance(m, nn.Linear):
            m.register_forward_hook(get_activation(f"{name}", feature_dict))

    return model


def extract_activations(loader, model):
    all_activations_dict = {}
    batch_activations_dict = {}
    registered_model = register_hooks(model, batch_activations_dict)

    for data in loader:
        registered_model(data.x, data.edge_index, data.batch)

        for key in batch_activations_dict.keys():
            if key not in all_activations_dict:
                all_activations_dict[key] = []

            activations_batch_data = batch_activations_dict[key].cpu().detach().numpy()
            all_activations_dict[key].append(activations_batch_data)

    for key in all_activations_dict.keys():
        all_activations_dict[key] = np.concatenate(all_activations_dict[key])


    return all_activations_dict