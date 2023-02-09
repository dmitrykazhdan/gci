import numpy as np

from sklearn.metrics import precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def compute_heuristic_performance(graph_heuristics, torchdrug_dataset):

    h_data = []
    y_data = []

    for graph_id in range(len(torchdrug_dataset.indices)):
        graph = torchdrug_dataset[graph_id]['graph']
        target = torchdrug_dataset[graph_id]['p_np']
        y_data.append(target)
        h_info = [graph_heuristic(graph) for graph_heuristic in graph_heuristics]
        h_data.append(h_info)

    h_data = np.array(h_data).astype(np.int)
    y_data = np.array(y_data)

    X_train, X_test, y_train, y_test = train_test_split(h_data, y_data, test_size = 0.33, random_state=2)

    clfs = [
        LogisticRegression(class_weight='auto'),
        DecisionTreeClassifier(),
        MLPClassifier()
    ]

    best_auroc, best_conf_matrix, best_f1, best_clf = -1, [], -1, None

    for clf in clfs:
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        aucroc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        if aucroc > best_auroc:
            best_auroc = aucroc
            best_clf = clf
            best_conf_matrix = conf_matrix
            best_f1 = f1

    print(f"Heuristic F1 score: {best_f1}")
    print(f"Heurisitc AUROC score: {best_auroc}")
    print(f"Heuristic conf. matrix: {best_conf_matrix}")

    return best_f1


def compute_heuristics_precisions(graph_heuristics, c_labels, torchdrug_dataset):
    n_heuristics = len(graph_heuristics)
    n_concepts = c_labels.shape[1]
    n_graphs = c_labels.shape[0]
    concept_heuristic_matrix = [[0 for _ in range(n_concepts)] for _ in range(n_heuristics)]

    h_labels = []
    c_one_hot_labels = []

    for graph_id in range(n_graphs):
        graph = torchdrug_dataset[graph_id]['graph']
        sample_c_one_hot = c_labels[graph_id]

        # Sample not part of the concept
        if -1 in list(sample_c_one_hot): continue

        heuristic_c = np.zeros(n_heuristics)

        for h_id, graph_heuristic in enumerate(graph_heuristics):
            if graph_heuristic(graph):
                heuristic_c[h_id] = 1

        c_one_hot_labels.append(sample_c_one_hot)
        h_labels.append(heuristic_c)

    c_one_hot_labels = np.array(c_one_hot_labels)
    h_labels = np.array(h_labels)

    for c_id in range(n_concepts):
        y_pred = c_one_hot_labels[:, c_id]
        for h_id in range(n_heuristics):
            y_true = h_labels[:, h_id]
            heuristic_precision = precision_score(y_true, y_pred)
            concept_heuristic_matrix[h_id][c_id] = heuristic_precision

    return concept_heuristic_matrix


def extract_heuristic_data(torchdrug_dataset, graph_heuristic):
    heuristic_data = []

    for graph_id in range(len(torchdrug_dataset.indices)):
        sample = torchdrug_dataset[graph_id]
        graph = sample.pop("graph")

        if graph_heuristic(graph):
            heuristic_data.append(1)
        else:
            heuristic_data.append(0)

    heuristic_data = np.array(heuristic_data)
    return heuristic_data


