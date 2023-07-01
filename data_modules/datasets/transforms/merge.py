from typing import Tuple

import numpy as np

from data_modules.datasets.transforms.json_serializable import JSONSerializable


class Merge(JSONSerializable, object):
    def __init__(self, merged_key: str, keys_to_merge: Tuple[str, str]):
        self.merged_key = merged_key
        assert len(keys_to_merge) == 2
        self.keys_to_merge = keys_to_merge

    def __call__(self, sample):
        merged_sequences = np.add(
            sample[self.keys_to_merge[0]].astype(object),
            sample[self.keys_to_merge[1]].astype(object),
        )
        sample[self.merged_key] = merged_sequences
        for key in self.keys_to_merge:
            sample.pop(key, None)
        return sample

    def to_json(self):
        return {"merged_key": self.merged_key, "keys_to_merge": self.keys_to_merge}
