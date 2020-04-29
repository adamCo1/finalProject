from relief import ReliefF


def __check_validity__(num_neighbors, num_features_to_keep):
    if num_neighbors <= 0:
        raise ValueError('number of neighbors must be greater than 0')
    if num_features_to_keep <= 0:
        raise ValueError('number of features to keep must be greater than 0')


class ReliefWrapper:

    def __init__(self, target_class, num_neighbors=100, num_features_to_keep=10):
        __check_validity__(num_neighbors, num_features_to_keep)
        self.num_neighbors = num_neighbors
        self.num_features_to_keep = num_features_to_keep
        self.alg = ReliefF(num_neighbors, num_features_to_keep)
        self.target_class = target_class

    def fit_data(self, features_vectors):
        if len(features_vectors.columns.values) > self.num_neighbors:
            raise ValueError('number of neighbors must be lower or equal to number of features')

        return self.alg.fit_transform(features_vectors, self.target_class)
