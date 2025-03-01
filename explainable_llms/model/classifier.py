import json
import numpy as np
from typing import Self, Sequence, Any, Mapping, Optional, Dict
from abc import ABC, abstractmethod


class Classifier(ABC):
    def __init__(
        self,
        n_inputs: int,
        precision: int = 4,
    ):
        self.n_inputs = n_inputs
        self.precision = precision

    @classmethod
    @abstractmethod
    def get_kwargs_from_example_instance(cls, instance: "Classifier") -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def predict(
        self, X: Sequence[Sequence[float]], noise_ratio: float = 0.0
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _get_explanation_list(
        self, x: Sequence[float], noise_ratio: float = 0.0
    ) -> list[list[Any]]:
        pass

    def get_explanation_json(self, x: Sequence[float], noise_ratio: float = 0.0) -> str:
        return json.dumps(
            self._get_explanation_list(x, noise_ratio=noise_ratio), ensure_ascii=False
        )

    @abstractmethod
    def get_reasoning_str(self, x: Sequence[float]) -> str:
        pass

    @abstractmethod
    def get_pseudocode(self, x: Optional[Sequence[float]] = None) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(**data)
