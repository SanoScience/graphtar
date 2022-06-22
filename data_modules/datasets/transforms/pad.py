import numpy as np

from data_modules.datasets.transforms.json_serializable import JSONSerializable


class Pad(JSONSerializable, object):

    def __init__(self, key_to_pad: str, target_length: int, padding_char: str):
        self.key_to_pad = key_to_pad
        self.target_length = target_length
        self.padding_char = padding_char

    def __call__(self, sample):
        assert self.target_length >= len(max(sample[self.key_to_pad], key=len))
        padding_func = np.vectorize(self.pad_array_element)

        sample[self.key_to_pad] = padding_func(sample[self.key_to_pad])
        return sample

    def pad_array_element(self, element):
        left_padding = (self.target_length - len(element)) // 2
        return element.rjust(self.target_length - left_padding, self.padding_char).ljust(self.target_length,
                                                                                         self.padding_char)

    def to_json(self):
        return {'key_to_pad': self.key_to_pad, 'target_length': self.target_length, 'padding_char': self.padding_char}
