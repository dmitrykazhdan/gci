from torch_geometric.loader import DataLoader

from experiments.synthetic_experiments.models.vanilla_gcn import GCN
from experiments.synthetic_experiments.datasets.ToySpuriousCorrelationsConceptDataset import ToySpuriousCorrelationsConceptDataset
from experiments.synthetic_experiments.model_training import run_train_loop
from concept_extraction.gcexplainer import *
from experiments.synthetic_experiments.activation_extraction_utils import *
from gci.visualisation import *
from gci.utils import *
from gci.metrics import *
from experiments.synthetic_experiments.config import *
from experiments.synthetic_experiments.synthetic_visualisation import *


def get_train_test_split(dataset, n_train):
    train_dataset, test_dataset = dataset[:n_train], dataset[n_train:]
    return train_dataset, test_dataset


def get_extractor_evaluation_model(num_node_features=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_channels, num_classes = 20, 2
    model = GCN(hidden_channels, num_node_features, num_classes).to(device)
    return model


############################################# HEURISTCS #############################################

def red_node_heuristic(graph):
    x_data = graph.x.cpu().detach().numpy()
    if float(x_data[0][0]) > 0.:
        return True
    else:
        return False

def blue_node_heuristic(graph):
    x_data = graph.x.cpu().detach().numpy()
    if float(x_data[0][0]) <= 0.:
        return True
    else:
        return False

def square_heuristic(graph):
    x_data = graph.x.cpu().detach().numpy()
    if x_data.shape[0] > 8:
        return True
    else:
        return False

def both_heuristics(graph):
    return square_heuristic(graph) and red_node_heuristic(graph)




def interpretation_alignment_experiment(visualise_dataset_samples=False):
    figs_path = os.path.join(FIG_PARENT_PATH, "interpretation_alignment")
    if os.path.exists(figs_path):
        shutil.rmtree(figs_path)
    os.mkdir(figs_path)

    # Get dataset
    n_graphs = 1000
    num_node_features = 2
    n_train = int(n_graphs * 0.8)
    n_test = n_graphs - n_train
    dataset = ToySpuriousCorrelationsConceptDataset(root="/tmp/myData", n_graphs=n_graphs, n_node_features=num_node_features)
    train_dataset, test_dataset = get_train_test_split(dataset, n_train=n_train)

    # Get model
    model = get_extractor_evaluation_model(num_node_features)

    # Optionally visualise a few dataset samples
    if visualise_dataset_samples:
        sample_ids = list(np.random.randint(low=0, high=100, size=8))
        dataset.visualise_samples(sample_ids)

    # Run basic model training loop
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    train_loader.n_samples = n_train
    test_loader.n_samples = n_test
    run_train_loop(model, train_loader, test_loader, n_epochs=10)

    # Extract model activations
    activations_dict = extract_activations(train_loader, model)
    model_activations = activations_dict['lin']

    # Define the heuristics to be used
    ground_truth_heuristics = [
        blue_node_heuristic,
        red_node_heuristic,
        square_heuristic,
        both_heuristics
    ]
    heuristic_names = ['blue nodes', 'red nodes', 'square attached', 'red nodes and square attached']

    # Compute the heuristic data
    h_data = compute_heuristic_data(ground_truth_heuristics, train_dataset)

    # Define the baselines
    coarse_explainer = GCExplainer(n_concepts=2)
    granular_explainer = GCExplainer(n_concepts=3)
    baselines = [coarse_explainer, granular_explainer]
    baseline_names = ['Coarse Explainer', 'Granular Explainer']

    for baseline, baseline_name in zip(baselines, baseline_names):
        baseline.train(model_activations)
        baseline_predictions = baseline.predict_concepts(model_activations)
        cat_baseline_predictions = convert_one_hot_arr_to_categorical(baseline_predictions)

        h_matrix = compute_heuristic_precision_matrix(baseline_predictions, h_data)
        print("="*20)
        print(h_matrix)
        print("\n")

        heuristic_matrix_path = os.path.join(figs_path, baseline_name + "_matrix" + ".png")
        plot_heuristic_matrix(h_matrix, baseline_predictions, heuristic_names, figpath=heuristic_matrix_path)

        concept_samples_path = os.path.join(figs_path, baseline_name + "_csamples" + ".png")
        plot_synthetic_baseline_concept_examples(cat_baseline_predictions, dataset, figpath=concept_samples_path, baseline_name=baseline_name)
