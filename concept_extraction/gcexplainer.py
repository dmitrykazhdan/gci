from sklearn.cluster import KMeans
import numpy as np

from concept_extraction.concept_extractor import ConceptDiscovery

class GCExplainer(ConceptDiscovery):
    def __init__(self, n_concepts=4):
        self.n_concepts=n_concepts

    def train(self, features_np):
        self.clf = KMeans(n_clusters=self.n_concepts)
        self.clf.fit(features_np)

    def predict_concepts(self, features_np):
        closest_cluster_labels = self.clf.predict(features_np)
        one_hot_labels = convert_to_one_hot(closest_cluster_labels, n_dims=self.n_concepts)
        return one_hot_labels



class NoisyGCExplainer(GCExplainer):
    def __init__(self, n_concepts=4, noise_ratio=0.5):
        super(NoisyGCExplainer, self).__init__(n_concepts)
        assert noise_ratio >= 0.0 and noise_ratio <= 1.0, "Please enter a float between 0.0 and 1.0"
        self.noise_ratio = noise_ratio

    def predict_concepts(self, features_np):
        n_samples = features_np.shape[0]
        n_noisy_samples = int(n_samples * self.noise_ratio)
        sample_indices = np.arange(n_samples)
        np.random.shuffle(sample_indices)
        noisy_sample_indices = sample_indices[:n_noisy_samples]

        noisy_sample_labels = np.random.randint(low=0, high=self.n_concepts, size=(n_noisy_samples))
        noisy_sample_labels_one_hot = convert_to_one_hot(noisy_sample_labels, n_dims=self.n_concepts)
        noisy_predictions = super().predict_concepts(features_np)
        noisy_predictions[noisy_sample_indices] = noisy_sample_labels_one_hot

        return noisy_predictions



class RandomExplainer(ConceptDiscovery):
    def __init__(self, n_concepts=4):
        self.n_concepts=n_concepts

    def train(self, features_np):
        pass

    def predict_concepts(self, features_np):
        n_samples = features_np.shape[0]
        random_labels = np.random.randint(low=0, high=self.n_concepts, size=(n_samples))
        random_labels_one_hot = convert_to_one_hot(random_labels, n_dims=self.n_concepts)
        return random_labels_one_hot


def convert_to_one_hot(cat_arr, n_dims=None):

    if n_dims is None: n_dims = np.max(cat_arr)+1

    one_hot_arr = np.zeros((cat_arr.size, n_dims))
    one_hot_arr[np.arange(cat_arr.size), cat_arr] = 1

    return one_hot_arr