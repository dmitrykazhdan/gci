import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_synthetic_baseline_concept_examples(cat_baseline_predictions, dataset, baseline_name="", figpath=None):
    n_concepts = np.max(cat_baseline_predictions) + 1
    n_samples_per_concept = 8

    fig = plt.figure(constrained_layout=True)
    c_subfigs = fig.subfigures(n_concepts, 1)

    for cid, c_subfig in enumerate(c_subfigs):
        c_sample_ids = np.where(cat_baseline_predictions == cid)[0]
        c_sample_ids = c_sample_ids[:n_samples_per_concept]
        c_axes = c_subfig.subplots(1, n_samples_per_concept)
        c_subfig.suptitle(f'{baseline_name} concept {cid} samples')

        for i, sampleid in enumerate(c_sample_ids):
            G = dataset.nx_graph_dict[sampleid]
            colors = dataset.nx_graph_colormaps[sampleid]
            nx.draw(G, ax=c_axes[i], node_size=20, node_color=colors)
            # axs[cid, i].set_title(f'C_{cid}, S_{i}')
            c_axes[i].margins(0.10)

    fig.tight_layout()
    plt.tight_layout()

    if figpath is not None:
        fig.savefig(figpath, bbox_inches='tight')

    plt.show()


