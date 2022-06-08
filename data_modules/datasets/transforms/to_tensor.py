from typing import Tuple

import torch


class ToTensor(object):
    def __init__(self, keys: Tuple[str, str]):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = torch.from_numpy(sample[key]).float()
        return sample
