import numpy as np
import torch
from torch_geometric.data import Data
import random
import networkx as nx
import matplotlib.pyplot as plt

from experiments.synthetic_experiments.datasets.ToyClassificationDataset import ToyGraphClassificationDataset



################################################################################################################################

class ToyCompletenessConceptDataset(ToyGraphClassificationDataset):
    def __init__(self, root, n_graphs=1000, n_node_features=1):
        super(ToyCompletenessConceptDataset, self).__init__(root=root, transform=None, pre_transform=None, pre_filter=None)
        self.name = 'ToyCompletenessDataset'
        self.n_graphs = n_graphs
        self.n_node_features = n_node_features
        self.nx_graph_dict = {}
        self.nx_graph_colormaps = {}
        self._generate_synthetic_graphs_with_concept_data()


    def _generate_synthetic_graphs_with_concept_data(self):
        n_graphs = self.n_graphs
        graphs_data_list = []

        for i in range(n_graphs):
            graph_data = self._generate_graph(graph_id=i)
            graphs_data_list.append(graph_data)

        self.data, self.slices = self.collate(graphs_data_list)


    def _generate_graph(self, graph_id):
        n_classes = 2
        class_label = random.randint(0, 1)
        graph = self._generate_random_graph_core()

        if class_label == 1:
            feature_offset = 0.2
            graph['features'] = graph['features'] + feature_offset

            n_nodes = graph['features'].shape[0]
            graph['colors'] = ['red' for _ in range(n_nodes)]


        add_spurious_signal = bool(random.uniform(0, 1.0) > 0.80)
        if add_spurious_signal:
            graph = self._attach_square_to_graph(graph)

        add_color_signal = bool(random.uniform(0, 1.0) > 0.80)
        if add_color_signal:
            graph = self._add_color_signal(graph)

        edge_list = graph['edges']
        nx_edgelist = [tuple([edge_list[0][i], edge_list[1][i]]) for i in range(edge_list.shape[1])]
        self.nx_graph_dict[graph_id] = nx.from_edgelist(nx_edgelist)
        self.nx_graph_colormaps[graph_id] = graph['colors']

        node_features = torch.from_numpy(graph['features']).to(torch.float)
        edge_index = torch.from_numpy(graph['edges'])
        ys = torch.from_numpy(np.array(class_label))

        graph_data = Data(x=node_features, edge_index=edge_index, y=ys)

        return graph_data


    def _generate_random_graph_core(self):
        n_nodes=10
        random_core = nx.barabasi_albert_graph(n=n_nodes, m=4)

        node_features = np.zeros((n_nodes, self.n_node_features))
        edge_list = nx.to_edgelist(random_core)
        edge_list = np.array([[i[0] for i in edge_list], [i[1] for i in edge_list]]).astype(np.int)
        return {"features": node_features, "edges": edge_list, "colors": ['blue' for _ in range(n_nodes)]}


    def _attach_square_to_graph(self, graph):
        node_features = graph['features']
        edge_list = graph['edges']

        n_nodes = node_features.shape[0]
        random_node_idx = random.randint(0, n_nodes-1)

        n_new_nodes = 4
        new_node_features = np.ones((n_new_nodes, self.n_node_features)) * node_features[0][0]
        new_node_edges = [[random_node_idx, n_nodes, n_nodes+1, n_nodes+2, n_nodes+3],
                          [n_nodes, n_nodes+1, n_nodes+2, n_nodes+3, n_nodes]]
        new_node_edges = np.array(new_node_edges)

        node_features = np.concatenate((node_features, new_node_features), axis=0)
        edge_list = np.concatenate((edge_list, new_node_edges), axis=-1)

        new_nodes_color = graph['colors'][0]
        return {"features": node_features, "edges": edge_list, "colors": graph['colors'] + [new_nodes_color for _ in range(n_new_nodes)]}


    def _add_color_signal(self, graph):
        node_features = graph['features']
        n_nodes = node_features.shape[0]
        random_node_idx = random.randint(0, n_nodes-1)

        graph['colors'][random_node_idx] = 'purple'
        graph['features'][random_node_idx] = graph['features'][random_node_idx]*0.0  - 0.2
        return graph


    def visualise_samples(self, graph_ids):
        for i, graph_id in enumerate(graph_ids):
            nx_graph = self.nx_graph_dict[graph_id]
            plt.figure(i)
            nx.draw(nx_graph, node_color=self.nx_graph_colormaps[graph_id])
        plt.show()

