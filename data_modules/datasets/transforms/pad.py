import numpy as np


class Pad(object):
    def __init__(self, target_length: int, key_to_pad: str, padding_char: str):
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
