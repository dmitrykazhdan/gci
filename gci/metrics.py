import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt


def measure_heuristic_completeness(h_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(h_data, y_data, test_size=0.33, random_state=2)

    clfs = [
        LogisticRegression(class_weight='auto'),
        DecisionTreeClassifier(),
        MLPClassifier()
    ]

    for clf in clfs:
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        print("F1 score is: ", f1)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion matrix is: ")
        print(conf_matrix)
        print("="*5)


def compute_heuristic_precision_matrix(c_one_hot_labels, h_labels):

    n_concepts = c_one_hot_labels.shape[1]
    n_heuristics = h_labels.shape[1]
    concept_heuristic_matrix = np.zeros((n_heuristics, n_concepts))

    for c_id in range(n_concepts):
        y_pred = c_one_hot_labels[:, c_id]
        for h_id in range(n_heuristics):
            y_true = h_labels[:, h_id]
            heuristic_precision = precision_score(y_true, y_pred)
            concept_heuristic_matrix[h_id][c_id] = heuristic_precision

    return concept_heuristic_matrix


def measure_heuristic_representation(h_data, activations_data, heuristic_names=None, figpath=None):
    n_heuristics = h_data.shape[1]

    f1_scores = []

    for hid in range(n_heuristics):
        print(f"HID: {hid}")
        hid_data = h_data[:, hid]

        X_train, X_test, y_train, y_test = train_test_split(activations_data, hid_data, test_size=0.33, random_state=2)

        clfs = [
            LogisticRegression(class_weight='auto'),
            DecisionTreeClassifier(),
            MLPClassifier()
        ]

        top_f1, top_matrix = -1, []

        for i in range(len(clfs)):
            clf = clfs[i].fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            if f1 > top_f1:
                top_matrix = conf_matrix
                top_f1 = f1

        print("F1 score is: ", top_f1)
        print("Confusion matrix is: ")
        print(top_matrix)
        print("=" * 5)
        f1_scores.append(top_f1)


    if heuristic_names is not None:
        plt.bar(heuristic_names, f1_scores)
        plt.title("Interpretation Predictability Scores")
        plt.xlabel("Interpretation Name")
        plt.ylabel("F1 Score Predictability")

        if figpath is not None:
            plt.savefig(figpath, bbox_inches='tight')

        plt.show()


def compute_heuristic_data(heuristics, graphs):

    n_graphs = len(graphs)
    n_heuristics = len(heuristics)
    h_data = np.zeros((n_graphs, n_heuristics)).astype(np.int)

    for gid, graph in enumerate(graphs):
        for hid, heuristic  in enumerate(heuristics):
            if heuristic(graph): h_data[gid][hid] = 1

    return h_data
