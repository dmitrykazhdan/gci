import os
import numpy as np
from torchdrug import data
import uuid
import matplotlib.pyplot as plt

from experiments.torchdrug_experiments.config import *


def plot_graphs_where_heuristic_is_true(torchdrug_graph_dataset, graph_heuristic, max_n_graphs_to_plot=30, n_rows=5, figpath=None):
    graphs_to_plot = []
    graph_ids = []

    for graph_id in range(len(torchdrug_graph_dataset.indices)):
        sample = torchdrug_graph_dataset[graph_id]
        graph = sample.pop("graph")

        if graph_heuristic(graph):
            graphs_to_plot.append(graph)
            graph_ids.append(graph_id)

        if len(graph_ids) >= max_n_graphs_to_plot:
            break

    graph = data.Molecule.pack(graphs_to_plot)
    graph.visualize(graph_ids, num_row=n_rows, save_file=figpath)



def plot_graphs_where_concept_is_true(torchdrug_dataset, c_labels, n_c_samples=10, n_rows=1, figpath=None):

    n_concepts = c_labels.shape[1]

    for c_id in range(n_concepts):
        c_sample_ids = np.where(c_labels[:, c_id] == 1)[0]
        graphs = []
        graph_labels = []

        for c_sample_id in c_sample_ids[:n_c_samples]:
            sample = torchdrug_dataset[c_sample_id]
            graph = sample.pop("graph")
            graphs.append(graph)
            graph_labels.append("")
            graph = data.Molecule.pack(graphs)
            graph.visualize(graph_labels, num_row=n_rows, save_file=figpath+f"c{c_id}.png")



def plot_graphs_where_heuristics_are_true(torchdrug_dataset, heuristics, heuristic_names, n_samples=5, figpath=None):

    graphss = [[] for _ in range(len(heuristics))]

    for graph_id in range(len(torchdrug_dataset.indices)):
        sample = torchdrug_dataset[graph_id]
        graph = sample.pop("graph")

        for hid, heuristic in enumerate(heuristics):
            if len(graphss[hid]) < n_samples and heuristic(graph):
                graphss[hid].append(graph)

    plot_torchdrug_graphs_grid(graphss, heuristic_names, save_file=figpath)



def plot_graphs_where_concept_is_true_one_fig(torchdrug_dataset, c_labels, n_c_samples=8, figpath=None):
    n_concepts = c_labels.shape[1]
    graphss = [[] for _ in range(n_concepts)]

    for c_id in range(n_concepts):
        c_sample_ids = np.where(c_labels[:, c_id] == 1)[0]
        graphs = []
        graph_labels = []

        for c_sample_id in c_sample_ids[:n_c_samples]:
            sample = torchdrug_dataset[c_sample_id]
            graph = sample.pop("graph")
            graphss[c_id].append(graph)

    titles = [f"GCExplainer Concept {i+1} samples" for i in range(n_concepts)]

    plot_torchdrug_graphs_grid(graphss, titles, save_file=figpath, n_cols=n_c_samples)



def plot_graphs_with_given_ids(dataset, graph_ids, n_rows=4, fig_path=None):
    graphs=[]

    for graph_id in graph_ids:
        sample = dataset[graph_id]
        graph = sample.pop("graph")
        graphs.append(graph)

    graph = data.Molecule.pack(graphs)

    if fig_path is None:
        fig_id = str(uuid.uuid4())
        fig_path = os.path.join(FIG_PATH_PREFIX, fig_id + ".png")

    graph.visualize(graph_ids, num_row=n_rows, save_file=fig_path)



def plot_first_n_graphs(dataset, n_samples_to_plot=10):
    graphs = []
    labels = []
    for i in range(n_samples_to_plot):
        sample = dataset[i]
        graphs.append(sample.pop("graph"))
        label = ["%s: %d" % (k, v) for k, v in sample.items()]
        label = ", ".join(label)
        labels.append(label)
    graph = data.Molecule.pack(graphs)
    graph.visualize(labels, num_row=1)



def plot_torchdrug_graphs_grid(graphss, row_titles, n_cols=None, save_file=None):

    n_rows = len(graphss)

    if n_cols is None:
        n_cols = len(graphss[0])

    figure_size = (3, 3)
    figure_size = (n_cols * figure_size[0], n_rows * figure_size[1])
    fig = plt.figure(figsize=figure_size)

    assert n_rows == len(row_titles), "Mismatching number of rows and row titles..."

    for row_id in range(n_rows):
        graphs = graphss[row_id]

        for i, graph in enumerate(graphs):
            ax = fig.add_subplot(n_rows, n_cols, (i + 1) + (row_id)*n_cols)
            if i > 0:
                title = ""
            else: title = row_titles[row_id]
            graph.visualize(title=title, ax=ax, atom_map=False)

    fig.tight_layout()

    if save_file:
        fig.savefig(save_file)


