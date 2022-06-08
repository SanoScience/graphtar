from typing import Dict
import numpy as np


class ToOneHot(object):
    def __init__(self, encoding_dict: Dict[str, int], key_to_encode: str):
        self.encoding_dict = encoding_dict
        self.key_to_encode = key_to_encode

    def __call__(self, sample):
        divided = np.array([list(word) for word in sample[self.key_to_encode]])
        encoded = np.array([self.one_hot_encode(word) for word in divided])
        sample[self.key_to_encode] = encoded.reshape(encoded.shape[0], -1)
        return sample

    def one_hot_encode(self, word):
        return [np.eye(max(self.encoding_dict.values()) + 1)[self.encoding_dict[char]] for char in word]