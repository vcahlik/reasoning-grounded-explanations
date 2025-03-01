import os
import shutil
from typing import Type, Mapping, Any, Sequence

from explainable_llms.model.decision_tree import DecisionTreeClassifier
from explainable_llms.model.logistic_regressor import LogisticRegressor
from explainable_llms.model.classifier import Classifier
from explainable_llms.dataset_generator import DatasetGenerator
from explainable_llms.problem import (
    ClassifierProblem,
    DecisionTreeProblem,
    LogisticRegressorProblem,
)


def _test_dataset_generator(
    classifier_class: Type[Classifier],
    problem_class: Type[ClassifierProblem[Any]],
    all_classifier_kwargs: Sequence[Mapping[str, Any]],
) -> None:
    base_output_dir = "pytest_temp_output"
    assert not os.path.exists(base_output_dir)
    dataset_generator = DatasetGenerator(classifier_class, problem_class)
    try:
        dataset_generator.generate_datasets(
            n_datasets=2,
            n_train_instances=10,
            n_test_instances=10,
            all_classifier_kwargs=all_classifier_kwargs,
            base_output_dir=base_output_dir,
        )
    finally:
        shutil.rmtree(base_output_dir)


def test_generate_logistic_regressor_datasets() -> None:
    all_classifier_kwargs = [
        {"n_inputs": 2},
        {"n_inputs": 5},
    ]
    _test_dataset_generator(
        LogisticRegressor, LogisticRegressorProblem, all_classifier_kwargs
    )


def test_generate_decision_tree_datasets() -> None:
    all_classifier_kwargs = [
        {"n_inputs": 2, "depth": 3},
        {"n_inputs": 5, "depth": 3},
    ]
    _test_dataset_generator(
        DecisionTreeClassifier, DecisionTreeProblem, all_classifier_kwargs
    )
