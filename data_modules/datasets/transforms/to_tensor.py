from typing import Tuple

import torch

from data_modules.datasets.transforms.json_serializable import JSONSerializable


class ToTensor(JSONSerializable, object):
    def __init__(self, keys: Tuple[str, str]):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = torch.from_numpy(sample[key]).float()
        return sample

    def to_json(self):
        return {'keys': self.keys}
