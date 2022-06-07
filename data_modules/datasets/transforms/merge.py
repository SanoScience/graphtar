from typing import Tuple

import numpy as np


class Merge(object):
    def __init__(self, merged_key: str, keys_to_merge: Tuple[str, str]):
        self.merged_key = merged_key
        assert len(keys_to_merge) == 2
        self.keys_to_merge = keys_to_merge

    def __call__(self, sample):
        merged_sequences = np.add(sample[self.keys_to_merge[0]], sample[self.keys_to_merge[1]])
        return {self.merged_key: merged_sequences, 'label': sample['label']}
