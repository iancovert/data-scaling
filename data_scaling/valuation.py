import numpy as np


def calculate_shapley_value(c, alpha, min_cardinality, max_cardinality):
    '''Calculate Shapley value from scaling law parameters.'''
    cardinalities = np.arange(min_cardinality, max_cardinality + 1)
    means = c * (cardinalities ** (- alpha))
    data_shapley = np.mean(means)
    return data_shapley
