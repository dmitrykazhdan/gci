from enum import Enum

N_MODEL_CONCEPTS = 4

class ModelTypes(str, Enum):
    VANILLA_GCN = "vanilla_gcn"
    CONCEPT_GNN = "concept_gnn"

FIG_PARENT_PATH = "~/Desktop/tmp"