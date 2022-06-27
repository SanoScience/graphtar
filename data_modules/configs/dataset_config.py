import sys
from json import JSONEncoder
from typing import TypedDict, Tuple, Dict

from torchvision.transforms.transforms import Compose

from data_modules.datasets.transforms.json_serializable import JSONSerializable

# noinspection PyUnresolvedReferences
from data_modules.datasets.transforms.merge import Merge
# noinspection PyUnresolvedReferences
from data_modules.datasets.transforms.pad import Pad
# noinspection PyUnresolvedReferences
from data_modules.datasets.transforms.to_one_hot import ToOneHot
# noinspection PyUnresolvedReferences
from data_modules.datasets.transforms.to_tensor import ToTensor


class DatasetConfig(TypedDict):
    csv_path: str
    batch_size: int
    train_val_ratio: Tuple[float, float]
    x_key: str
    y_key: str
    transform: Compose


class DatasetConfigEncoder(JSONEncoder):

    def default(self, obj):
        if isinstance(obj, int) or isinstance(obj, str):
            return obj
        if isinstance(obj, Compose):
            return self.encode_transforms(obj)

    def encode_transforms(self, value: Compose):
        transforms = []

        for transform in value.transforms:
            if isinstance(transform, JSONSerializable):
                transforms.append({type(transform).__name__: transform.to_json()})

        return transforms


class DatasetConfigDecoder:
    @staticmethod
    def object_hook(obj):
        if 'transform' in obj.keys():
            obj['transform'] = DatasetConfigDecoder.deserialize_transforms(obj['transform'])
        return obj

    @staticmethod
    def deserialize_transforms(transforms):
        transforms_initialized = []
        for transform in transforms:
            transform_key = list(transform.keys())[0]
            cls: JSONSerializable = getattr(sys.modules[__name__], transform_key)
            transforms_initialized.append(DatasetConfigDecoder.from_json(cls, transform[transform_key]))
        transform = Compose(transforms_initialized)
        return transform

    @staticmethod
    def from_json(cls, dict: Dict):
        return cls(**dict)
