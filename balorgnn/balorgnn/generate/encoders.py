from enum import Enum

import numpy as np

import sklearn.preprocessing as sk

from math import log2

class OneHotEncoder():
    def encode(self, input):
        # encoder expects N samples with M features as a array with shape (N, M) to one hot encode
        # so input needs to be NP array with shape (1, 1) 
        input = np.array(input).reshape(1, 1)

        # output is also 2D, but I want just the encoding
        output = self.encoder.transform(input)
        return output[0]


    def fit(self):
        self.encoder = sk.OneHotEncoder(sparse_output=False, handle_unknown='error')

        # encoder expects N samples with M features as a array with shape (N, M) to one hot encode
        # so list needs to be NP array with (N, 1), e.g. a single feature, but with one sample for each possible value of the feature 
        tags = np.array(self.tags).reshape(-1, 1)

        self.encoder.fit(tags)

class NormalizedEncoder():
    def __init__(self):
        self.max = None
    def normalize(self, value):
        assert(self.max is not None)
        map_0_to_1 = value/self.max
        map_neg_1_to_1 = (map_0_to_1 * 2) - 1

        # convert to 1D NP array to match the output of the onehotencoder
        return np.array([map_neg_1_to_1])

    def set_max(self, max):
        self.max = float(max)

class LogNormalizedEncoder():
    def __init__(self):
        self.max = None
        self.bias = None
    def normalize(self, value):
        assert(self.max is not None)
        assert(self.bias is not None)

        scale_factor = log2(self.max + self.bias) - log2(self.bias)
        log_value = log2(value + self.bias) - log2(self.bias)

        log_value_0_to_1 = log_value / scale_factor
        log_value_neg_1_to_1 = (log_value_0_to_1 * 2) - 1

        # convert to 1D NP array to match the output of the onehotencoder
        return np.array([log_value_neg_1_to_1])

    def set_max(self, max):
        self.max = float(max)
    def set_bias(self, bias):
        self.bias = float(bias)


class EncoderMethod(Enum):
    ONE_HOT = 0
    NORMALIZED = 1
    LOG_NORMALIZED = 2

class EncoderType(Enum):
    NODE = 0
    EDGE = 1