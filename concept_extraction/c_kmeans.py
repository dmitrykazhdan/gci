from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE


def extract_concepts_for_target_class(targets, features_np, target_class_id=None, n_concepts=4, tsne_preprocess=False):
    c_labels = np.ones((features_np.shape[0], n_concepts)) * -1

    if target_class_id is not None:
        cls_sample_ids = np.where(targets == target_class_id)[0]
    else:
        cls_sample_ids = np.arange(targets.shape[0])

    cls_features_np = features_np[cls_sample_ids]
    c_class_labels, _ = extract_concepts_kmeans(cls_features_np, n_concepts=n_concepts, tsne_preprocess=tsne_preprocess)

    for c_id in range(n_concepts):
        c_sample_ids = np.where(c_class_labels == c_id)[0]
        c_sample_ids = cls_sample_ids[c_sample_ids]
        c_labels[c_sample_ids, :] = 0
        c_labels[c_sample_ids, c_id] = 1

    c_labels = c_labels.astype(np.int)

    return c_labels


def extract_concepts_kmeans(features_np, n_concepts=4, tsne_preprocess=False):

    if tsne_preprocess:
        tsne = TSNE(n_components=2)
        features_np = tsne.fit_transform(features_np)

    kmeans = KMeans(n_clusters=n_concepts, random_state=0).fit(features_np)

    c_labels = kmeans.labels_
    kmeans_dists = np.abs(kmeans.fit_transform(features_np))
    c_filtered_ids = np.arange(features_np.shape[0])
    c_labels = c_labels.astype(np.int)
    return c_labels, c_filtered_ids
