from abc import ABC, abstractmethod


class JSONSerializable(ABC):
    @abstractmethod
    def to_json(self):
        pass
