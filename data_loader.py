import pandas as pd
from sklearn.preprocessing import LabelEncoder
'''
    class for cleaning the data before using it
'''
class data_processing:
    def __init__(self,dataset_path,sep,number_of_bins):
        self.dataset = pd.read_csv(dataset_path,sep=sep)
        self.number_of_bins = number_of_bins

    def clear_na(self):
        self.dataset.dropna(inplace=True)

    def caterogical_to_numeric(self):
        le = LabelEncoder()
        self.dataset[self.dataset.columns[-1]] = le.fit_transform(self.dataset[self.dataset.columns[-1]])

    def discritsize_columns(self,cols_to_discrit="all"):
        interval_labels = []
        for i in range(1, self.number_of_bins + 1):
            interval_labels.append(i)
        if cols_to_discrit == "all":
            for col_name in self.dataset.columns[:-1].values:
                self.dataset[col_name] = pd.qcut(self.dataset[col_name], q=self.number_of_bins, labels= interval_labels)
        else:
            for col_name in cols_to_discrit:
                self.dataset[col_name] = pd.qcut(self.dataset[col_name], q=self.number_of_bins, labels= interval_labels)

    def prepare_data(self):
        self.clear_na()
        self.caterogical_to_numeric()
        self.discritsize_columns()
        self.base_features = self.dataset[self.dataset.columns[1:-1]]
        self.classes = self.dataset[self.dataset.columns[-1]]
        self.features_names = list(self.dataset.columns[1:-1].values)
        self.class_name = self.dataset.columns[-1]
        return self.base_features, self.classes, self.features_names, self.class_name