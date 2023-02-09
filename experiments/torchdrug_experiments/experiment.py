import os.path
import tkinter
import matplotlib
matplotlib.use('TkAgg')

from torchdrug import datasets

from concept_extraction.c_kmeans import *

from experiments.torchdrug_experiments.models.torchdrug_gcn import TorchDrugGCN
from experiments.torchdrug_experiments.torchdrug_model_loader import load_torchdrug_model
from experiments.torchdrug_experiments.heuristic_encodings import *
from gci.torchdrug.torchdrug_activation_extraction import *
from gci.torchdrug.torchdrug_metrics import *
from gci.visualisation import *
from gci.torchdrug.torchdrug_visualisation import *

from gci.utils import *

from experiments.torchdrug_experiments.config import FIG_PATH_PREFIX


def explore_torchdrug_dataset():
    dataset = datasets.BBBP("~/molecule-datasets/")
    plot_first_n_graphs(dataset, n_samples_to_plot=5)


def get_torchdrug_model(model_type, dataset, train_set, valid_set, test_set):

    device = torch.device("cpu")

    model_path = f"./torch_drug_{model_type.value}.pth"
    num_node_features = dataset.node_feature_dim
    hidden_dims = [256, 256, 128, 64]
    readout = "sum"

    if model_type == ModelTypes.VANILLA_GCN:
        model = TorchDrugGCN(input_dim=num_node_features, hidden_dims=hidden_dims, readout=readout).to(device)
    else:
        raise ValueError(f"Unrecognised model type: {model_type}")

    load_torchdrug_model(model_path=model_path, model=model, train_set=train_set,
                         valid_set=valid_set, test_set=test_set, dataset=dataset)

    return model

def run_torchdrug_dataset_experiment():

    ######################################## Param setup  ########################################

    dataset = datasets.BBBP("~/molecule-datasets/") # BBBP dataset info: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#bbbp-datasets
    figs_path = os.path.join(FIG_PATH_PREFIX, "bbbp")
    model_to_run = ModelTypes.VANILLA_GCN

    plot_explainer_activations = False
    plot_explainer_samples = True
    plot_heuristic_samples = False

    heuristics = [has_hydroxyl, has_ketone, has_aromatic_ring, has_phenyl,
                  has_chloro, has_fluoro, contains_carboxyl_group]
    heuristic_names = ['Hydroxyl', 'Ketone', 'Aromatic Ring', 'Phenyl',
                       'Chlorine', 'Fluorine', 'Carboxyl']


    ######################################## Experiment Execution  ########################################

    setup_folder_from_scratch(figs_path)

    train_set, valid_set, test_set = generate_or_load_torch_train_test_split(dataset, n_train=int(0.8*len(dataset)), n_val=int(0.1*len(dataset)))

    model = get_torchdrug_model(model_to_run, dataset, train_set, valid_set, test_set)

    activations, targets = extract_activations_from_torchdrug_model(model, train_set)

    for n_concepts in [10]:
        for target_class_id in [0]:
            train_concept_labels = extract_concepts_for_target_class(target_class_id=target_class_id, targets=targets, features_np=activations, n_concepts=n_concepts)

            if plot_explainer_samples:
                plot_graphs_where_concept_is_true_one_fig(train_set, train_concept_labels, figpath=os.path.join(figs_path, f"gcexplainer_samples_{target_class_id}.png"))

            if plot_explainer_activations:
                plot_activations_tsne(activations, labels_np=remove_c_padding(train_concept_labels, n_concepts))

            if plot_heuristic_samples:
                plot_graphs_where_heuristics_are_true(train_set, heuristics=heuristics, heuristic_names=heuristic_names,
                                                      n_samples=8, figpath=os.path.join(figs_path, f"heuristic_samples_{target_class_id}.png"))

            ia_matrix = compute_heuristics_precisions(graph_heuristics=heuristics, c_labels=train_concept_labels, torchdrug_dataset=train_set)
            for row in ia_matrix:
                print([round(i, 2) for i in row])

            plot_heuristic_matrix(ia_matrix, train_concept_labels, heuristic_names, figpath=os.path.join(figs_path, f'GCExplainer_matrix_cls_{target_class_id}_c{n_concepts}.png'))

        compute_heuristic_performance(graph_heuristics=heuristics, torchdrug_dataset=train_set)

    print("Ran successfully...")



def remove_c_padding(c_labels, n_concepts):
    categorical_c_labels = []
    for i in c_labels:
        if i[0] == -1:
            categorical_c_labels.append(n_concepts)
        else:
            categorical_c_labels.append(list(i).index(1))

    return categorical_c_labels



def iterate_and_show_all_atom_feature_types(torchdrug_dataset):

    all_atom_features = []

    for graph_id in range(len(torchdrug_dataset.indices)):
        sample = torchdrug_dataset[graph_id]
        graph = sample.pop("graph")

        atom_features = graph.atom_feature.cpu().detach().numpy()
        atom_types = graph.atom_type.cpu().detach().numpy()

        for i, atom_feature in enumerate(atom_features):
            atom_type = atom_types[i]
            if atom_type == 8:
                all_atom_features.append(tuple(atom_feature))

    all_atom_features = set(all_atom_features)

    atom_feature_examples = {k: 0 for k in all_atom_features}


    for graph_id in range(len(torchdrug_dataset.indices)):
        sample = torchdrug_dataset[graph_id]
        graph = sample.pop("graph")

        atom_features = graph.atom_feature.cpu().detach().numpy()
        atom_types = graph.atom_type.cpu().detach().numpy()

        for i, atom_feature in enumerate(atom_features):
            atom_type = atom_types[i]
            if atom_type == 8:
                f = tuple(atom_feature)
                atom_feature_examples[f] += 1

    sorted_examples = {k: v for k, v in sorted(atom_feature_examples.items(), key=lambda item: item[1])}

    return sorted_examples




