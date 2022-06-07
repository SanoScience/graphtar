from typing import Dict

import numpy as np


class ToOneHot(object):
    def __init__(self, encoding_dict: Dict[str, int], key_to_encode: str):
        self.encoding_dict = encoding_dict
        self.key_to_encode = key_to_encode
        self.char_encoding_func = np.vectorize(
            lambda x: np.eye(max(self.encoding_dict.values()) + 1)[self.encoding_dict[x]].astype(object))

    def __call__(self, sample):
        division_func = np.vectorize(self.divide_string)
        divided = division_func(sample[self.key_to_encode])
        array_encoding_func = np.vectorize(self.one_hot_encode)
        encoded = array_encoding_func(divided)
        sample[self.key_to_encode] = encoded
        return sample

    def divide_string(self, string):
        return np.array(list(string), dtype=object)

    def one_hot_encode(self, string):
        return self.char_encoding_func(string)
