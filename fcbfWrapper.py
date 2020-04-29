from fcbf import fcbf

def validate_params(threshhold):
    if threshhold < 0 or threshhold > 1:
        raise ValueError('threshhold must be between 0 and 1')

class FcbfWrapper:

    def __init__(self, class_vector, threshhold):
        self.threshhold = threshhold
        self.class_vector = class_vector


    def fit_data(self, data):
        fcbf_alg = fcbf(self.threshhold)
        return fcbf_alg.fcbf(data, self.class_vector)