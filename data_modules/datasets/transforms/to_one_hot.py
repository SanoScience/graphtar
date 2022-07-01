from typing import Dict

import numpy as np

from data_modules.datasets.transforms.json_serializable import JSONSerializable


class ToOneHot(JSONSerializable, object):
    def __init__(self, encoding_dict: Dict[str, int], key_to_encode: str):
        self.encoding_dict = encoding_dict
        self.key_to_encode = key_to_encode

    def __call__(self, sample):
        divided = np.array([word for word in sample[self.key_to_encode]])
        sample[self.key_to_encode] = np.array([self.one_hot_encode(word) for word in divided])
        return sample

    def one_hot_encode(self, word):
        return np.array([np.eye(max(self.encoding_dict.values()) + 1)[self.encoding_dict[char]] for char in word])

    def to_json(self):
        return {'encoding_dict': self.encoding_dict, 'key_to_encode': self.key_to_encode}
