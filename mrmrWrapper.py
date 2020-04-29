import pymrmr as mrmr


class MrmrWrapper:

    def __init__(self, type, num_features_to_return):
        self.__check_validity__(type, num_features_to_return)
        self.type = type
        self.num_features_to_return = num_features_to_return

    def __check_validity__(self, type, num):
        if type not in ["MID", "MIQ"]:
            raise TypeError('type of MRMR must be MID or MIQ')
        if num <= 0:
            raise TypeError('number of returned features must be greater than 0')

    def fit_data(self, data):
        selected_features = mrmr.mRMR(data, self.type, self.num_features_to_return)
        return data[selected_features]