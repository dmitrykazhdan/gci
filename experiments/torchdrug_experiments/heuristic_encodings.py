import numpy as np
from torchdrug import data

from experiments.torchdrug_experiments.encoding_mappings import *


def has_hydroxyl(graph):
    '''
    Here, we define a hydroxyl group as having an OH atom.
    In case of TorchDrug, these are oxygen-type atoms with a single bond.
    '''
    atom_features = graph.atom_feature.cpu().detach().numpy()
    atom_types = graph.atom_type.cpu().detach().numpy()
    edge_list = graph.edge_list.cpu().detach().numpy()

    for i, atom_type in enumerate(atom_types):
        if atom_type == OXYGEN:
            node_edges = [e for e in edge_list if e[0]==i]

            if len(node_edges) == 1 and node_edges[0][2] == 0:
                return True

    return False



def has_ketone(graph):
    '''
    Here, we define a ketone group as:
        - An oxygen atom
        - Double-bonded to a carbon atom
        - Which has at least 2 other carbon bonds to other structures
    '''
    atom_features = graph.atom_feature.cpu().detach().numpy()
    atom_types = graph.atom_type.cpu().detach().numpy()
    edge_list = graph.edge_list.cpu().detach().numpy()

    for i, atom_type in enumerate(atom_types):
        if atom_type == OXYGEN:
            node_edges = [e for e in edge_list if e[0]==i]

            if len(node_edges) == 1 and node_edges[0][2] == 1:
                adjacent_node_id = node_edges[0][1]

                if atom_types[adjacent_node_id] == CARBON:
                    carbon_edges =  [e for e in edge_list if e[0]==adjacent_node_id]

                    if len(carbon_edges) == 3:
                        return True

    return False



def has_aromatic_ring(graph):
    '''
    Here we make the simplifying assumption of what an aromatic ring is:
        - Carbon atom
        - With at least 2 other connections with aromatic bonds
    '''
    atom_features = graph.atom_feature.cpu().detach().numpy()
    atom_types = graph.atom_type.cpu().detach().numpy()
    edge_list = graph.edge_list.cpu().detach().numpy()

    for i, atom_type in enumerate(atom_types):
        if atom_type == CARBON:
            node_edges = [e for e in edge_list if e[0]==i]

            if len([e for e in node_edges if e[2] == 3]) >= 2:
                return True

    return False


def has_phenyl(graph):
    atom_features = graph.atom_feature.cpu().detach().numpy()
    atom_types = graph.atom_type.cpu().detach().numpy()
    edge_list = graph.edge_list.cpu().detach().numpy()

    for atom_id, atom_type in enumerate(atom_types):
        if atom_type == CARBON:
            node_edges = [e for e in edge_list if e[0]==atom_id]

            if len(node_edges) != 3: continue

            n_single_bonds = len([i for i in node_edges if i[2] == 0])
            n_aromatic_bonds = len([i for i in node_edges if i[2] == 3])

            if not (n_single_bonds == 1 and n_aromatic_bonds == 2): continue

            # return True

            if node_edges[0][2] == 0:
                starting_node = node_edges[1][1]
            else:
                starting_node = node_edges[0][1]

            passed_nodes = [atom_id]
            ring_complete = False
            while True:
                starting_node_edges = [e for e in edge_list if e[0]==starting_node]

                if len(starting_node_edges) != 2: break

                untraversed_edges = [e for e in starting_node_edges if e[1] not in passed_nodes]

                if len(untraversed_edges) == 0:
                    if len(passed_nodes) == 5: ring_complete = True
                    break

                elif len(untraversed_edges) == 1:
                    passed_nodes.append(starting_node)
                    starting_node = untraversed_edges[0][1]
                else:
                    break

            if ring_complete: return True

    return False



def has_chloro(graph):
    '''
    Has a chlorine atom attached to the molecule
    '''
    atom_features = graph.atom_feature.cpu().detach().numpy()
    atom_types = graph.atom_type.cpu().detach().numpy()
    edge_list = graph.edge_list.cpu().detach().numpy()

    for atom_id, atom_type in enumerate(atom_types):
        if atom_type == CHLORINE:
            node_edges = [e for e in edge_list if e[0] == atom_id]

            if len(node_edges) == 1: return True

    return False


def has_fluoro(graph):
    atom_features = graph.atom_feature.cpu().detach().numpy()
    atom_types = graph.atom_type.cpu().detach().numpy()
    edge_list = graph.edge_list.cpu().detach().numpy()

    for atom_id, atom_type in enumerate(atom_types):
        if atom_type == FLUORINE:
            node_edges = [e for e in edge_list if e[0] == atom_id]

            if len(node_edges) == 1: return True

    return False


def has_nitro(graph):
    atom_features = graph.atom_feature.cpu().detach().numpy()
    atom_types = graph.atom_type.cpu().detach().numpy()
    edge_list = graph.edge_list.cpu().detach().numpy()

    for atom_id, atom_type in enumerate(atom_types):
        if atom_type == NITROGEN:
            return True
    return False


def contains_carboxyl_group(graph):

    atom_types =  graph.atom_type.cpu().detach().numpy().astype(int)
    edge_list = graph.edge_list.cpu().detach().numpy()

    for atom_id, atom_type in enumerate(atom_types):
        if atom_type == OXYGEN:
            node_edges = [e for e in edge_list if e[0] == atom_id]

            if len(node_edges) != 1 or node_edges[0][2] != DOUBLE_BOND: continue

            connecting_atom_id = node_edges[0][1]

            if atom_types[connecting_atom_id] != CARBON: continue
            connecting_node_edges = [e for e in edge_list if e[0] == connecting_atom_id and e[1] != atom_id]

            if len(connecting_node_edges) != 2: continue

            for connecting_node_edge in connecting_node_edges:
                if connecting_node_edge[2] == SINGLE_BOND and atom_types[connecting_node_edge[1]] == OXYGEN \
                        and len([e for e in edge_list if e[0] == connecting_node_edge[1]]) == 1:
                    return True

    return False