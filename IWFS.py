# import pyitlib as metrics
from data_loader import data_processing
import pandas as pd
from pyitlib import discrete_random_variable as metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
import copy


'''
@params variable1 and variable2 are numpy array of feature/class values
'''

def Symmetric_uncertainty(X, Y):
    return 2*metrics.information_mutual(X,Y)/(metrics.entropy_residual(X) + metrics.entropy_residual(Y))


def Interaction_Weight(X, Y, Z):
    interaction_gain = metrics.information_mutual_conditional(X,Y,Z) - metrics.information_mutual(X,Y)
    return (interaction_gain/(metrics.entropy_residual(X) + metrics.entropy_residual(Y)))+1



'''
Input:
            1.size of feature subset output : int
            2.dataset : pandas dataframe, including all features and class variables
Output:
            A subset of features: list of strings
 '''

''' TODO add a classifier to find the threshold to stop the searching'''
class IWFS:

    def __init__(self, features_vectors, class_vector, num_of_features, features_names, class_name):
        self.features_vectors = features_vectors
        self.class_vector = class_vector
        self.total_num_of_features = int(num_of_features)
        self.selected_features = []
        self.unselected_features = features_names
        self.num_of_selected_features = 0
        self.class_name = class_name
        self.features_names = features_names
        self.features_weights = dict()
        self.feautre_and_class_symmetric_uncertainty = dict()
        for name in self.features_names:
            self.features_weights.update({name:1})
            self.feautre_and_class_symmetric_uncertainty.update({name:0})

    def fit_data(self, data, output_features_subset_size):
        for feature_name in self.features_names:
            self.feautre_and_class_symmetric_uncertainty[feature_name] = Symmetric_uncertainty(self.features_vectors[feature_name].values.astype(int),self.class_vector.values.astype(int))
        while len(self.selected_features) < output_features_subset_size:
            most_relevant_feature = None
            current_max_relevance_score = float('-inf')
            for candidate_feature in self.unselected_features:
                adjusted_relevance_measure_score = self.features_weights[candidate_feature] * (1 + self.feautre_and_class_symmetric_uncertainty[candidate_feature])
                # self.feature_and_class_adjusted_relevance_measure[candidate_feature] = adjusted_relevance_measure_score
                if most_relevant_feature is None or current_max_relevance_score < adjusted_relevance_measure_score:
                    most_relevant_feature = candidate_feature
                    current_max_relevance_score = adjusted_relevance_measure_score
            self.selected_features.append(most_relevant_feature)
            self.unselected_features.remove(most_relevant_feature)
            for candidate_feature in self.unselected_features:
                if candidate_feature != most_relevant_feature:
                    candidate_feature_interaction_weight = Interaction_Weight(self.features_vectors[most_relevant_feature].values.astype(int), self.features_vectors[candidate_feature].values.astype(int), self.class_vector.values.astype(int))
                    self.features_weights[candidate_feature] = self.features_weights[candidate_feature] * candidate_feature_interaction_weight
        return self.selected_features

class IwfsWrapper:
    def __init__(self, class_vector, num_of_features, features_names, class_name):
        self.classifier = tree.DecisionTreeClassifier()
        self.class_vector = class_vector
        self.features_names = features_names
        self.num_of_features = num_of_features
        self.class_name = class_name
        self.curr_best_set = None
        self.curr_best_score = 0

    def fit_data(self, data):
        print("Starting IWFS feature selection...")
        selector = IWFS(data, self.class_vector, self.num_of_features, self.features_names, self.class_name)

        for i in range(0,len(self.features_names)):
            curr_subset_features_names = selector.fit_data(data, i+1)
            # treeModel = self.classifier.fit(self.features_vectors[curr_subset_features_names], self.class_vector)
            cross_val_results = cross_val_score(self.classifier, data[curr_subset_features_names].values.astype(int), self.class_vector.values.astype(int), cv=10)
            set_score = sum(cross_val_results)/len(cross_val_results)
            if self.curr_best_score < set_score:
                self.curr_best_set = copy.deepcopy(curr_subset_features_names)
                self.curr_best_score = set_score

        print("best features subset: %s length %d"%(str(self.curr_best_set),len(self.curr_best_set)))
        return data[self.curr_best_set]
