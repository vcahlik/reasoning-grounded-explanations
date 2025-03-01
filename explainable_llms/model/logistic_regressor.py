import random
import numpy as np
from typing import Optional, Sequence, Any, Dict
from functools import cache

from explainable_llms.model.classifier import Classifier


class LogisticRegressor(Classifier):
    MAX_WEIGHT = 10

    def __init__(
        self,
        n_inputs: int,
        precision: int = 4,
        weights: Optional[Sequence[Any]] = None,
    ):
        super().__init__(n_inputs, precision)
        self.weights = weights or self._generate_weights()

    @classmethod
    def get_kwargs_from_example_instance(cls, instance: "Classifier") -> Dict[str, Any]:
        return {
            "n_inputs": instance.n_inputs,
            "precision": instance.precision,
        }

    @property
    def name(self) -> str:
        return "logistic regressor"

    def _generate_weights(self) -> Sequence[Any]:
        return [
            round(random.uniform(-1 * self.MAX_WEIGHT, self.MAX_WEIGHT), self.precision)
            for _ in range(self.n_inputs)
        ]

    def predict(
        self, X: Sequence[Sequence[float]], noise_ratio: float = 0.0
    ) -> np.ndarray:
        Y = []
        for x in X:
            if noise_ratio > 0 and random.random() <= noise_ratio:
                Y.append(random.randint(0, 1))
                continue
            y = 0
            for weight, value in zip(self.weights, x):
                y += weight * value
            Y.append(int(y > 0))
        return np.array(Y)

    def _get_explanation_list(
        self, x: Sequence[float], noise_ratio: float = 0.0
    ) -> list[list[Any]]:
        explanation: list[list[Any]] = []
        y = 0
        for i, (weight, value) in enumerate(zip(self.weights, x)):
            w_times_x = round(weight * value, self.precision)
            new_y = round(y + w_times_x, self.precision)
            explanation.append(
                [
                    i,
                    # f"{weight} * {value} = {w_times_x}",
                    f"w[{i}] * x[{i}] = {w_times_x}",
                    # f"{y} {'+' if w_times_x >= 0 else '-'} {abs(w_times_x)} = {new_y}",
                    f"y {'+' if w_times_x >= 0 else '-'} {abs(w_times_x)} = {new_y}",
                ]
            )
            y = new_y
        if noise_ratio > 0 and random.random() <= noise_ratio:
            y = random.randint(0, 1)
        else:
            y = int(y > 0)
        explanation.append(["OUTPUT", y])
        return explanation

    def get_reasoning_str(self, x: Sequence[float]) -> str:
        explanation_list = self._get_explanation_list(x)
        choice_strings = []
        for choice in explanation_list[:-1]:
            choice_strings.append(" ".join([x.split(" ")[-1] for x in choice[1:]]))
        choice_strings.append(str(explanation_list[-1][-1]))
        return ";".join(choice_strings)

    @cache
    def get_pseudocode(self, x: Optional[Sequence[float]] = None) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_inputs": self.n_inputs,
            "precision": self.precision,
            "weights": self.weights,
        }
