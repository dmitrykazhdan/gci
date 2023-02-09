import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

from gci.utils import *


def plot_heuristic_matrix(matrix, c_labels, heuristic_names, figpath=None, show_plot=False):
    plt.clf()
    matrix = np.array(matrix)
    plt.imshow(matrix, interpolation='None', cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xticks(np.arange(c_labels.shape[1]), [f'c_{i}' for i in range(c_labels.shape[1])])
    plt.yticks(np.arange(len(heuristic_names)), heuristic_names)
    plt.title("Interpretation-Alignment Matrix")
    plt.tight_layout()

    if figpath is not None:
        plt.savefig(figpath, bbox_inches='tight')

    if show_plot:
        plt.show()


def visualise_heuristic_data(h_data, tsne_features, model_activations):
    for i in range(h_data.shape[1]):
        cat_h_data = convert_one_hot_arr_to_categorical(h_data[:, i:i+1])
        plot_tsne_features(tsne_features, cat_h_data)
        plot_activations_tsne(model_activations, cat_h_data)


def plot_activations_tsne(features_np, labels_np=None, n_components=2):
    if labels_np is None:
        labels_np = np.ones(shape=features_np.shape[0])

    tsne = TSNE(n_components=n_components)
    tsne_result = tsne.fit_transform(features_np)
    plot_tsne_features(tsne_result, labels_np=labels_np)


def plot_tsne_features(tsne_features, labels_np=None):
    if labels_np is None:
        labels_np = np.ones(shape=tsne_features.shape[0])

    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels_np)
    handles, labels = scatter.legend_elements()
    plt.legend(handles, labels, loc="upper right", title="Sizes")
    plt.show()
