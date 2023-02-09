import numpy as np
from torchdrug import data

def extract_activations_from_torchdrug_model(torchdrug_model, torchdrug_dataset, n_batches=None):
    torchdrug_model.eval()
    activations = []
    targets = []
    dataloader = data.DataLoader(torchdrug_dataset, batch_size=32, shuffle=False)

    for i, batch in enumerate(dataloader):
        graph = batch["graph"]
        output = torchdrug_model(graph, graph.node_feature.float())
        graph_labels = batch['p_np'].cpu().detach().numpy()

        features = output["graph_feature"]
        targets.append(graph_labels)

        activations.append(features)

        if n_batches is not None and i > n_batches: break

    activations = [pred.cpu().detach().numpy() for pred in activations]
    activations = np.concatenate(activations)
    targets = np.concatenate(targets)

    return activations, targets
