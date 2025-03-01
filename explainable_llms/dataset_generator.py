import json
import os
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from typing import Sequence, Any, Mapping, Optional, Type

from explainable_llms.model.classifier import Classifier
from explainable_llms.problem import ClassifierProblem


class DatasetGenerator:
    def __init__(
        self,
        classifier_class: Type[Classifier],
        problem_class: Type[ClassifierProblem[Any]],
        classifier_kwargs: Optional[Mapping[str, Any]] = None,
        classifier_example: Optional[Classifier] = None,
    ):
        self.classifier_class = classifier_class
        self.problem_class = problem_class
        self.classifier_kwargs = classifier_kwargs
        self.classifier_example = classifier_example

    def generate_dataset(
        self,
        classifier: Classifier,
        problem: ClassifierProblem[Any],
        n_examples: int,
    ) -> Dataset:
        xs = []
        decisions = []
        for _ in range(n_examples):
            x = problem.generate_xs(n_examples=1)[0]
            decision = str(classifier.predict([x])[0])
            xs.append(x)
            decisions.append(decision)
        return Dataset.from_dict({"x": xs, "decision": decisions})

    def generate_dataset_and_problem(self, n_examples: int) -> Dataset:
        if self.classifier_kwargs is not None:
            assert self.classifier_example is None
            classifier_kwargs = self.classifier_kwargs
        else:
            assert self.classifier_example is not None
            classifier_kwargs = self.classifier_class.get_kwargs_from_example_instance(
                self.classifier_example
            )
        classifier = self.classifier_class(**classifier_kwargs)
        problem = self.problem_class(classifier)
        return self.generate_dataset(classifier, problem, n_examples), problem

    def generate_datasets(
        self,
        n_datasets: int,
        n_train_instances: int,
        n_test_instances: int,
        all_classifier_kwargs: Sequence[Mapping[str, Any]],
        base_output_dir: str,
    ) -> None:
        if self.classifier_kwargs is not None:
            raise RuntimeError(
                "This method can't be run when the generator has classifier_kwargs configured as they would be ignored"
            )
        os.makedirs(base_output_dir, exist_ok=True)
        for classifier_kwargs in tqdm(all_classifier_kwargs):
            output_dir = f"{base_output_dir}/" + "_".join(
                [f"{key}_{value}" for key, value in classifier_kwargs.items()]
            )
            os.makedirs(output_dir, exist_ok=False)
            for i in tqdm(range(n_datasets)):
                classifier = self.classifier_class(**classifier_kwargs)
                problem = self.problem_class(classifier)
                dataset = DatasetDict(
                    {
                        "train": self.generate_dataset(
                            classifier, problem, n_train_instances
                        ),
                        "test": self.generate_dataset(
                            classifier, problem, n_test_instances
                        ),
                    }
                )
                dataset.save_to_disk(f"{output_dir}/dataset_{i}")
                with open(
                    f"{output_dir}/classifier_{i}.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(classifier.to_dict(), f, ensure_ascii=False)
