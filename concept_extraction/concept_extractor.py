from abc import ABC, abstractmethod


class ConceptDiscovery(ABC):

    @abstractmethod
    def train(self, features_np):
        '''
        :param features_np: numpy array of shape (n_samples, n_features)
        '''
        pass

    @abstractmethod
    def predict_concepts(self, features_np):
        '''
        :param features_np: numpy array of shape (n_samples, n_features)
        :return: binary concept labels of the form (n_samples, n_concepts)
        '''
        pass