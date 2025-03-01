import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from typing import Optional, Sequence, Any, Dict
from functools import cache

from explainable_llms.model.classifier import Classifier


class DecisionTreeClassifier(Classifier):
    def __init__(
        self,
        n_inputs: int,
        depth: int = 3,
        precision: int = 4,
        structure: Optional[Sequence[Any]] = None,
    ):
        super().__init__(n_inputs, precision)
        self.depth = depth
        self.structure = structure or self._generate_structure(depth=depth)

    @classmethod
    def get_kwargs_from_example_instance(cls, instance: "Classifier") -> Dict[str, Any]:
        if not isinstance(instance, cls):
            raise ValueError(f"The instance must be of {cls} class")
        return {
            "n_inputs": instance.n_inputs,
            "depth": instance.depth,
            "precision": instance.precision,
        }

    @property
    def name(self) -> str:
        return "decision tree classifier"

    def _generate_structure(
        self,
        depth: int,
        last_variable: Optional[int] = None,
        ranges: Optional[Sequence[Sequence[float]]] = None,
    ) -> Sequence[Any]:
        if ranges is None:
            ranges = [[0, 1]] * self.n_inputs
        else:
            ranges = list(ranges)
        valid_variables = list(range(0, self.n_inputs))
        if last_variable is not None:
            valid_variables.remove(last_variable)
        variable = random.choice(valid_variables)
        operator = "lt" if random.randint(0, 1) == 0 else "gt"
        buffer = (ranges[variable][1] - ranges[variable][0]) / 4
        separator = round(
            np.random.uniform(
                ranges[variable][0] + buffer, ranges[variable][1] - buffer
            ),
            self.precision,
        )
        ranges_lt = ranges.copy()
        ranges_lt[variable] = [ranges[variable][0], separator]
        ranges_gt = ranges.copy()
        ranges_gt[variable] = [separator, ranges[variable][1]]
        return [
            variable,
            operator,
            separator,
            self._generate_structure(
                depth - 1,
                last_variable=variable,
                ranges=(ranges_lt if operator == "lt" else ranges_gt),
            )
            if depth > 0
            else 1,
            self._generate_structure(
                depth - 1,
                last_variable=variable,
                ranges=(ranges_gt if operator == "lt" else ranges_lt),
            )
            if depth > 0
            else 0,
        ]

    def predict(
        self, X: Sequence[Sequence[float]], noise_ratio: float = 0.0
    ) -> np.ndarray:
        Y = []
        for x in X:
            if noise_ratio > 0 and random.random() <= noise_ratio:
                Y.append(random.randint(0, 1))
                continue
            structure = self.structure
            while not isinstance(structure, int):
                (variable, operator, separator, left_structure, right_structure) = (
                    structure
                )
                if (
                    (operator == "gt" and x[variable] > separator)
                    or operator == "lt"
                    and x[variable] < separator
                ):
                    structure = left_structure
                else:
                    structure = right_structure
            Y.append(structure)
        return np.array(Y)

    def _get_explanation_list(
        self, x: Sequence[float], noise_ratio: float = 0.0
    ) -> list[list[Any]]:
        explanation = []
        structure = self.structure
        while not isinstance(structure, int):
            (variable, operator, separator, left_structure, right_structure) = structure
            if (
                (operator == "gt" and x[variable] > separator)
                or operator == "lt"
                and x[variable] < separator
            ):
                structure = left_structure
                truth = True
            else:
                structure = right_structure
                truth = False
            explanation.append(
                [f"{x[variable]} {'>' if operator == 'gt' else '<'} {separator}", truth]
            )
        if noise_ratio > 0 and random.random() <= noise_ratio:
            y = random.randint(0, 1)
        else:
            y = structure
        explanation.append(["OUTPUT", y])
        return explanation

    def get_reasoning_str(self, x: Sequence[float]) -> str:
        explanation_list = self._get_explanation_list(x)
        return ",".join([str(int(choice[-1])) for choice in explanation_list])

    @cache
    def get_pseudocode(self, x: Optional[tuple[float]] = None) -> str:
        input_str = str(list(x)) if x else "x"
        pseudocode = "def decision_tree(x):\n"
        pseudocode += self._get_pseudocode(self.structure, "    ")
        pseudocode += f"\nprint(decision_tree({input_str}))"
        return pseudocode

    @staticmethod
    def _get_pseudocode(structure: Sequence[Any], indent: str) -> str:
        if isinstance(structure, int):
            return f"{indent}return {structure}\n"
        pseudocode = ""
        (variable, operator, separator, left_structure, right_structure) = structure
        operator = ">" if operator == "gt" else "<"
        pseudocode += f"{indent}if x[{variable}] {operator} {separator}:\n"
        pseudocode += DecisionTreeClassifier._get_pseudocode(
            left_structure, indent + "    "
        )
        pseudocode += f"{indent}else:\n"
        pseudocode += DecisionTreeClassifier._get_pseudocode(
            right_structure, indent + "    "
        )
        return pseudocode

    def plot(self, show: bool = True, ax: Optional[plt.Axes] = None) -> None:
        if self.n_inputs != 2:
            raise NotImplementedError(
                "Tree can't be plotted as it does not have 2 input variables"
            )
        feature_1, feature_2 = np.meshgrid(
            np.linspace(0, 1),
            np.linspace(0, 1),
        )
        grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
        y_pred = -1 * np.reshape(self.predict(grid), feature_1.shape)
        display = DecisionBoundaryDisplay(xx0=feature_1, xx1=feature_2, response=y_pred)
        cmap = "Pastel1"
        if ax is not None:
            display.plot(ax=ax, cmap=cmap)
        else:
            display.plot(cmap=cmap)
            plt.axis("equal")
        if show:
            plt.show()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_inputs": self.n_inputs,
            "precision": self.precision,
            "depth": self.depth,
            "structure": self.structure,
        }
