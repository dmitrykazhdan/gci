import numpy as np
import torch
from torch_geometric.data import Data
import random
import networkx as nx
import matplotlib.pyplot as plt

from experiments.synthetic_experiments.datasets.ToyClassificationDataset import ToyGraphClassificationDataset


class ToyConceptsDataset(ToyGraphClassificationDataset):
    def __init__(self, root, n_graphs=1000, n_node_features=1):
        super(ToyConceptsDataset, self).__init__(root=root, transform=None, pre_transform=None, pre_filter=None)
        self.name = 'ToyConceptsDataset'
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
            np_features = graph['features']
            n_nodes = np_features.shape[0]
            is_triangle = bool(random.randint(0, 1))
            if is_triangle:
                graph['features'] = np.ones((n_nodes, self.n_node_features))
                graph['colors'] = ['yellow' for _ in range(n_nodes)]
            else:
                graph['features'] = np.ones((n_nodes, self.n_node_features)) * 2
                graph['colors'] = ['red' for _ in range(n_nodes)]

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
        n_nodes=8
        random_core = nx.barabasi_albert_graph(n=n_nodes, m=4)

        node_features = np.zeros((n_nodes, self.n_node_features))
        edge_list = nx.to_edgelist(random_core)
        edge_list = np.array([[i[0] for i in edge_list], [i[1] for i in edge_list]]).astype(np.int)
        return {"features": node_features, "edges": edge_list, "colors": ['blue' for _ in range(n_nodes)]}

    def visualise_samples(self, graph_ids):
        for i, graph_id in enumerate(graph_ids):
            nx_graph = self.nx_graph_dict[graph_id]
            plt.figure(i)
            nx.draw(nx_graph, node_color=self.nx_graph_colormaps[graph_id])
        plt.show()
