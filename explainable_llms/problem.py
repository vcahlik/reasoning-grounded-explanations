from abc import ABC, abstractmethod
import json
import re
import random
from datasets import Dataset
from typing import (
    Mapping,
    Any,
    Sequence,
    TypeVar,
    Generic,
    Optional,
    Callable,
    TYPE_CHECKING,
)

from explainable_llms.model.classifier import Classifier
from explainable_llms.model.decision_tree import DecisionTreeClassifier
from explainable_llms.model.logistic_regressor import LogisticRegressor
from explainable_llms.utils import DatasetType

if TYPE_CHECKING:
    from explainable_llms.dataset_generator import DatasetGenerator


T = TypeVar("T", bound=Classifier)


class Problem(ABC):
    @staticmethod
    @abstractmethod
    def get_human_input(example: Mapping[str, Any]) -> str:
        pass

    @staticmethod
    def get_decision(example: Mapping[str, Any]) -> str:
        return str(example["decision"])

    def get_explanation(self, example: Mapping[str, Any]) -> str:
        return str(example["explanation"])

    def get_reasoning(self, example: Mapping[str, Any]) -> str:
        return str(example["reasoning"])

    def notify_build_next_finetuning_dataset_instance(
        self,
        example_dataset_mapping_fn: Callable[
            [Mapping[str, Any], DatasetType, "Problem"], Mapping[str, Any]
        ],
        dataset_type: DatasetType,
    ) -> None:
        pass

    @abstractmethod
    def get_few_shot_decision_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
        reveal_classifier_type: bool = True,
    ) -> str:
        pass

    @abstractmethod
    def get_few_shot_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        pass

    @abstractmethod
    def get_structural_decision_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        pass

    @abstractmethod
    def get_structural_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        pass

    @staticmethod
    @abstractmethod
    def parse_decision_target(example: Mapping[str, Any]) -> int:
        pass

    @staticmethod
    @abstractmethod
    def parse_decision_output(example: Mapping[str, Any]) -> int:
        pass

    @staticmethod
    @abstractmethod
    def parse_explanation_decision_output(example: Mapping[str, Any]) -> int:
        pass

    @staticmethod
    def _parse_reasoning(reasoning: str) -> str:
        return json.dumps([int(x) for x in reasoning if x != ","])

    @classmethod
    def parse_reasoning_target(cls, example: Mapping[str, Any]) -> str:
        return cls._parse_reasoning(example["reasoning_target"])

    @classmethod
    def parse_reasoning_output(cls, example: Mapping[str, Any]) -> str:
        return cls._parse_reasoning(example["reasoning_output"])

    @abstractmethod
    def generate_xs(self, n_examples: int) -> list[list[float]]:
        pass


class ClassifierProblem(Problem, Generic[T]):
    MIN_X_VALUE = 0.0
    MAX_X_VALUE = 1.0

    def __init__(self, classifier: T):
        self.classifier = classifier

    @staticmethod
    def get_human_input(example: Mapping[str, Any]) -> str:
        x = example["x"]
        return f"X: {x}"

    def get_explanation(self, example: Mapping[str, Any]) -> str:
        x = example["x"]
        return str(self.classifier.get_explanation_json(x))

    def get_reasoning(self, example: Mapping[str, Any]) -> str:
        x = example["x"]
        return self.classifier.get_reasoning_str(x)

    def get_few_shot_decision_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
        reveal_classifier_type: bool = True,
    ) -> str:
        if example_xs is None:
            raise NotImplementedError
        x = example["x"]
        examples_str = "\n\n".join(
            [
                f"Response for x={x}:\n{self.classifier.predict([x], noise_ratio=noise_ratio)[0]}"
                for x in example_xs
            ]
        )
        classifier_name = (
            self.classifier.name if reveal_classifier_type else "classifier"
        )
        return f"""You are a {classifier_name} inference engine that performs inference based on examples of inference. Use the examples to understand the structure and behavior of the {classifier_name} and use this knowledge to produce the correct response.
Your response must be in the form of a single number (0 or 1) and nothing else!

Examples:

{examples_str}

Your response will be machine processed! Do not output anything except 0 or 1! Your output MUST be a parsable integer!

Response for x={x}:
"""

    def get_structural_decision_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        if example_xs is None:
            raise NotImplementedError
        x = example["x"]
        examples_str = ""
        if example_xs:
            examples_str = "\n\n".join(
                [
                    f"Response for x={x}:\n{self.classifier.predict([x], noise_ratio=noise_ratio)[0]}"
                    for x in example_xs
                ]
            )
        return f"""You are a code executor that executes code and returns the output of the code execution.
Your response must be in the form of a single number (0 or 1) and nothing else!

Response example:
1

Code:
{self.classifier.get_pseudocode()}

{examples_str}

Your response will be machine processed! Do not output anything except 0 or 1! Your output MUST be a parsable integer!

Response for x={x}:
"""

    @staticmethod
    def parse_decision_target(example: Mapping[str, Any]) -> int:
        return int(example["decision_target"])

    @staticmethod
    def parse_decision_output(example: Mapping[str, Any]) -> int:
        decision_output = example["decision_output"]
        decision_output = int(decision_output.strip()[0])
        assert decision_output in (0, 1)
        return decision_output  # type: ignore

    @staticmethod
    def parse_explanation_decision_output(example: Mapping[str, Any]) -> int:
        explanation_output = example["explanation_output"]
        pattern = r"```json\s(.*?)```"
        match = re.search(pattern, explanation_output, re.DOTALL)
        if match is not None:
            explanation_output = match[1]
        explanation_output_list = json.loads(explanation_output)
        explanation_decision_output = explanation_output_list[-1][1]
        assert explanation_decision_output in (0, 1)
        return explanation_decision_output  # type: ignore

    def generate_xs(self, n_examples: int) -> list[list[float]]:
        return [
            [
                round(
                    random.uniform(self.MIN_X_VALUE, self.MAX_X_VALUE),
                    self.classifier.precision,
                )
                for _ in range(self.classifier.n_inputs)
            ]
            for _ in range(n_examples)
        ]


class FinetunedICLProblem(Problem):
    def __init__(
        self,
        dataset_generator: "DatasetGenerator",
        test_problem: ClassifierProblem[Any],
        n_examples: int,
    ):
        self.dataset_generator = dataset_generator
        self.test_problem = test_problem
        self.n_few_shot_examples = n_examples
        self._current_dataset: Optional[Dataset] = None
        self._current_problem: Optional[Problem] = None

    def notify_build_next_finetuning_dataset_instance(
        self,
        example_dataset_mapping_fn: Callable[
            [Mapping[str, Any], DatasetType, Problem], Mapping[str, Any]
        ],
        dataset_type: DatasetType,
    ) -> None:
        if dataset_type == DatasetType.TRAIN:
            self._current_dataset, self._current_problem = (
                self.dataset_generator.generate_dataset_and_problem(
                    self.n_few_shot_examples
                )
            )
        elif dataset_type == DatasetType.TEST:
            self._current_dataset = self.dataset_generator.generate_dataset(
                self.test_problem.classifier,
                self.test_problem,
                self.n_few_shot_examples,
            )
            self._current_problem = self.test_problem
        else:
            raise ValueError(f"Invalid dataset type {dataset_type}")
        self._current_dataset = self._current_dataset.map(
            example_dataset_mapping_fn,
            fn_kwargs={
                "dataset_type": dataset_type,
                "problem": self._current_problem,
            },
        )

    def get_human_input(self, example: Mapping[str, Any]) -> str:  # type: ignore
        if self._current_dataset is None or self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        human_input = ""
        for ex in self._current_dataset:
            human_input += self._current_problem.get_human_input(ex)
            # human_input += "\n" + self.get_reasoning(ex)
            human_input += "\nANSWER: " + self.get_decision(ex)
            human_input += "\nEXPLANATION: " + self.get_explanation(ex)
            human_input += "\n\n"
        human_input += self._current_problem.get_human_input(example)
        return human_input

    def get_decision(self, example: Mapping[str, Any]) -> str:  # type: ignore
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.get_decision(example)

    def get_explanation(self, example: Mapping[str, Any]) -> str:
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.get_explanation(example)

    def get_reasoning(self, example: Mapping[str, Any]) -> str:
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.get_reasoning(example)

    def get_few_shot_decision_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
        reveal_classifier_type: bool = True,
    ) -> str:
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.get_few_shot_decision_prompt(
            example=example,
            example_xs=example_xs,
            noise_ratio=noise_ratio,
            reveal_classifier_type=reveal_classifier_type,
        )

    def get_few_shot_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.get_few_shot_explanation_prompt(
            example=example,
            example_xs=example_xs,
            noise_ratio=noise_ratio,
        )

    def get_structural_decision_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.get_structural_decision_prompt(
            example=example,
            example_xs=example_xs,
            noise_ratio=noise_ratio,
        )

    def get_structural_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.get_structural_explanation_prompt(
            example=example,
            example_xs=example_xs,
            noise_ratio=noise_ratio,
        )

    def parse_decision_target(self, example: Mapping[str, Any]) -> int:  # type: ignore
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.parse_decision_target(example)

    def parse_decision_output(self, example: Mapping[str, Any]) -> int:  # type: ignore
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.parse_decision_output(example)

    def parse_explanation_decision_output(self, example: Mapping[str, Any]) -> int:  # type: ignore
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.parse_explanation_decision_output(example)

    def _parse_reasoning(self, reasoning: str) -> str:  # type: ignore
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem._parse_reasoning(reasoning)

    def parse_reasoning_target(self, example: Mapping[str, Any]) -> str:  # type: ignore
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.parse_reasoning_target(example)

    def parse_reasoning_output(self, example: Mapping[str, Any]) -> str:  # type: ignore
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.parse_reasoning_output(example)

    def generate_xs(self, n_examples: int) -> list[list[float]]:
        if self._current_problem is None:
            raise RuntimeError("example_dataset has not been generated")
        return self._current_problem.generate_xs(n_examples)


class DecisionTreeProblem(ClassifierProblem[DecisionTreeClassifier]):
    def __init__(self, decision_tree: DecisionTreeClassifier):
        super().__init__(classifier=decision_tree)

    def get_few_shot_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        if example_xs is None:
            raise NotImplementedError
        x = example["x"]
        examples_str = "\n\n".join(
            [
                f"JSON response for x={x}:\n{self.classifier.get_explanation_json(x, noise_ratio=noise_ratio)}"
                for x in example_xs
            ]
        )
        classifier_name = self.classifier.name
        return f"""You are a {classifier_name} inference engine that performs inference based on examples of inference. Use the examples to understand the structure and behavior of the {classifier_name} and use this knowledge to produce the correct response.
Return a JSON list of decision tree branch decisions that will be evaluated for the GIVEN VALUE OF X, together with the final output. You must deduce the comparison operators and branch thresholds from the inference examples. In your response, there must be exactly {self.classifier.depth + 1} branch decision(s) followed by the final output!
Your response must be in the form of a JSON list with the following structure:
[["x[INDEX] COMPARISON_OPERATOR BRANCH_THRESHOLD", COMPARISON_RESULT], ..., ["x[INDEX] COMPARISON_OPERATOR BRANCH_THRESHOLD", COMPARISON_RESULT], ["OUTPUT", OUTPUT]]
Remember to properly enclose the JSON list in square brackets!

{examples_str}

Your response will be machine processed! Do not output anything except the JSON! Your output MUST be parsable JSON!

JSON response for x={x}:
"""

    def get_structural_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        if example_xs is None:
            raise NotImplementedError
        x = example["x"]
        examples_str = ""
        if example_xs:
            examples_str = "\n\n".join(
                [
                    f"JSON response example (for x={x}):\n{self.classifier.get_explanation_json(x, noise_ratio=noise_ratio)}"
                    for x in example_xs
                ]
            )
        return f"""You are an analyzer of decision tree inference. Analyze the code and return a JSON list of decision tree branch decisions that will be evaluated DURING INFERENCE for the GIVEN VALUE OF X, together with the final output.
DO NOT output the list of all of the decision tree branch rules! Only output the rules for the branches that will be evaluated for the given value of x!
Your response must be in the form of a JSON list with the following structure:
[["x[INDEX]", OPERATOR, THRESHOLD, true/false], ..., ["x[INDEX]", OPERATOR, THRESHOLD, true/false], ["OUTPUT", 0/1]]
Remember to properly enclose the JSON list in square brackets!

Code:
{self.classifier.get_pseudocode()}

{examples_str}

Your response will be machine processed! Do not output anything except the JSON! Your output MUST be parsable JSON!

JSON response for x={x}:
"""


class LogisticRegressorProblem(ClassifierProblem[LogisticRegressor]):
    MIN_X_VALUE = -1.0
    MAX_X_VALUE = 1.0

    def __init__(self, logistic_regressor: LogisticRegressor):
        super().__init__(classifier=logistic_regressor)

    @staticmethod
    def _parse_reasoning(reasoning: str) -> str:
        choices = reasoning.split(";")
        parsed_reasoning = [
            float(number) for numbers in choices[:-1] for number in numbers.split()
        ]
        parsed_reasoning.append(int(choices[-1]))
        return json.dumps(parsed_reasoning)

    def get_few_shot_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        if example_xs is None:
            raise NotImplementedError
        x = example["x"]
        examples_str = "\n\n".join(
            [
                f"JSON response for x={x}:\n{self.classifier.get_explanation_json(x, noise_ratio=noise_ratio)}"
                for x in example_xs
            ]
        )
        classifier_name = self.classifier.name
        return f"""You are a {classifier_name} inference engine that performs inference based on examples of inference. Use the examples to understand the structure and behavior of the {classifier_name} and use this knowledge to produce the correct response.
Return a JSON list of partial calculations that will be evaluated for the GIVEN VALUE OF X (one partial calculation for each dimension), together with the final output. In your response, there must be exactly {self.classifier.n_inputs} sublists with calculations, followed by the final output!
Your response must be in the form of a JSON list with the following structure:
[["W[0] * X[0] = WX[0]", "0 + WX[0] = Y[0]"], ["W[1] * X[1] = WX[1]", "Y[0] + WX[1] = Y[1]"], ..., ["W[{self.classifier.n_inputs - 1}] * X[{self.classifier.n_inputs - 1}] = WX[{self.classifier.n_inputs - 1}]", "Y[{self.classifier.n_inputs - 2}] + WX[{self.classifier.n_inputs - 1}] = Y[{self.classifier.n_inputs - 1}]"], ["OUTPUT", OUTPUT]]
Remember to properly enclose the JSON list in square brackets!

{examples_str}

Your response will be machine processed! Do not output anything except the JSON! Your output MUST be parsable JSON!

JSON response for x={x}:
"""

    def get_structural_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        if example_xs is None:
            raise NotImplementedError
        x = example["x"]
        examples_str = ""
        if example_xs:
            examples_str = "\n\n".join(
                [
                    f"JSON response example (for x={x}):\n{self.classifier.get_explanation_json(x, noise_ratio=noise_ratio)}"
                    for x in example_xs
                ]
            )
        return f"""You are an analyzer of logistic regressor inference. Analyze the code and return a JSON list of partial calculations that will be evaluated DURING INFERENCE for the GIVEN VALUE OF X (one partial calculation for each dimension), together with the final output.
Your response must be in the form of a JSON list with the following structure:
[["W[0] * X[0] = WX[0]", "0 + WX[0] = Y[0]"], ["W[1] * X[1] = WX[1]", "Y[0] + WX[1] = Y[1]"], ..., ["W[{self.classifier.n_inputs - 1}] * X[{self.classifier.n_inputs - 1}] = WX[{self.classifier.n_inputs - 1}]", "Y[{self.classifier.n_inputs - 2}] + WX[{self.classifier.n_inputs - 1}] = Y[{self.classifier.n_inputs - 1}]"], ["OUTPUT", OUTPUT]]
Remember to properly enclose the JSON list in square brackets!

Code:
{self.classifier.get_pseudocode()}

{examples_str}

Your response will be machine processed! Do not output anything except the JSON! Your output MUST be parsable JSON!

JSON response for x={x}:
"""


class NLPDecisionTreeProblem(Problem):
    def __init__(self, example_dataset: Optional[Dataset] = None):
        self.example_dataset = example_dataset

    @staticmethod
    def get_human_input(example: Mapping[str, Any]) -> str:
        return str(example["x"])

    @staticmethod
    def parse_decision_target(example: Mapping[str, Any]) -> int:
        decision_target = example["decision_target"]
        if "issued" in decision_target:
            decision_target = 1
        elif "denied" in decision_target:
            decision_target = 0
        else:
            raise ValueError
        return decision_target  # type: ignore

    @staticmethod
    def parse_decision_output(example: Mapping[str, Any]) -> int:
        decision_output = example["decision_output"]
        if "issued" in decision_output:
            decision_output = 1
        elif "denied" in decision_output:
            decision_output = 0
        else:
            raise ValueError
        return decision_output  # type: ignore

    @staticmethod
    def parse_explanation_decision_output(example: Mapping[str, Any]) -> int:
        explanation_output = example["explanation_output"]
        explanation_decision_output = explanation_output.split(". ")[-1]
        if "issued" in explanation_decision_output:
            explanation_decision_output = 1
        elif "denied" in explanation_decision_output:
            explanation_decision_output = 0
        else:
            raise ValueError
        return explanation_decision_output  # type: ignore

    def generate_xs(self, n_examples: int) -> list[list[float]]:
        raise NotImplementedError

    def get_few_shot_decision_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
        reveal_classifier_type: bool = True,
    ) -> str:
        if example_xs is not None or self.example_dataset is None:
            raise NotImplementedError
        if noise_ratio > 0:
            raise NotImplementedError
        x = example["x"]
        examples_str = "\n\n".join(
            [
                f"Response for x={ex['x']}:\n{ex['decision_target']}"
                for ex in self.example_dataset
            ]
        )
        classifier_name = "decision tree" if reveal_classifier_type else "classifier"
        return f"""You are a {classifier_name} inference engine that performs inference based on examples of inference. Use the examples to understand the structure and behavior of the {classifier_name} and use this knowledge to produce the correct response.
Your response must be exactly one of the following strings (without the quotes): "The mortgage is issued." or "The mortgage is denied."

Examples:

{examples_str}

Your response will be machine processed! Do not output any comments besides the decision string!

Response for x={x}:
"""

    def get_few_shot_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        if example_xs is not None or self.example_dataset is None:
            raise NotImplementedError
        if noise_ratio > 0:
            raise NotImplementedError
        x = example["x"]
        examples_str = "\n\n".join(
            [
                f"JSON response for x={ex['x']}:\n{ex['explanation_target']}"
                for ex in self.example_dataset
            ]
        )
        return f"""You are a decision tree inference engine that performs inference based on examples of inference. Use the examples to understand the structure and behavior of the decision tree and use this knowledge to produce the correct response.
Return a JSON list of decision tree branch decisions that will be evaluated for the GIVEN VALUE OF X, together with the final decision. You must deduce the comparison operators and branch thresholds from the inference examples. In your response, state the branch decision(s) followed by the final output!
Your final decision must be exactly one of the following strings (without the quotes): "Therefore, the mortgage is issued." or "Therefore, the mortgage is denied."

{examples_str}

Your response will be machine processed! Follow exactly the format of the examples! Do not output any comments!

JSON response for x={x}:
"""

    def get_structural_decision_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        raise NotImplementedError

    def get_structural_explanation_prompt(
        self,
        example: Mapping[str, Any],
        example_xs: Optional[Sequence[Sequence[float]]],
        noise_ratio: float = 0.0,
    ) -> str:
        raise NotImplementedError
