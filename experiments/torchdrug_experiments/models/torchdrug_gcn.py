from collections.abc import Sequence
import torch
from torch import nn
from torchdrug import core, layers


class TorchDrugGCN(nn.Module, core.Configurable):
    '''
    Largely based on the following example: https://torchdrug.ai/docs/_modules/torchdrug/models/gcn.html#GraphConvolutionalNetwork
    '''

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(TorchDrugGCN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GraphConv(self.dims[i], self.dims[i + 1], edge_input_dim, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)


    def forward(self, graph, input, all_loss=None, metric=None):

        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }

