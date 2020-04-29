
from sklearn import tree
from sklearn.model_selection import cross_val_score

'''
@params
X pandas data frame of all features
Y pandas series class vector
features_names names of the features used to train and classify  
'''
class Decision_Tree:
    def __init__(self, X, Y, features_names, class_name):
        self.features_vectors = X[features_names]
        self.class_vector = Y
        self.features_names = features_names
        self.class_name = class_name
        self.classifier = tree.DecisionTreeClassifier()

    def get_mean_accuarcy_by_cross_validation(self, cross_validations=10):
        cross_val_results = cross_val_score(self.classifier, self.features_vectors[self.features_names].values.astype(int), self.class_vector.values.astype(int), cv=10)
        score = sum(cross_val_results)/len(cross_val_results)
        return score
