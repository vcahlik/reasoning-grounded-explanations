from datasets import load_from_disk, disable_caching, Dataset
from datasets.utils.logging import disable_progress_bar
from unsloth import FastLanguageModel
import simplejson as json
import os
import argparse
from typing import Sequence, Optional, TextIO, Type, Any
import dataclasses
import sys
import datetime
from dataclasses import dataclass

from explainable_llms.training import (
    formatting_prompts_func,
    build_finetuning_dataset,
    train_llm,
    add_batch_inference,
    parse_outputs,
    ResponseMode,
)
from explainable_llms.evaluation import calculate_metrics
from explainable_llms.llm import (
    get_llm_config,
    DummyLLM,
    load_pretrained_llm,
    convert_llm_to_peft,
)
from explainable_llms.model.classifier import Classifier
from explainable_llms.utils import DatasetType
from explainable_llms.model.decision_tree import DecisionTreeClassifier
from explainable_llms.model.logistic_regressor import LogisticRegressor
from explainable_llms.dataset_generator import DatasetGenerator
from explainable_llms.problem import (
    Problem,
    ClassifierProblem,
    DecisionTreeProblem,
    LogisticRegressorProblem,
    NLPDecisionTreeProblem,
    FinetunedICLProblem,
)


disable_progress_bar()
disable_caching()


@dataclass
class ExperimentConfig:
    model_name: str
    problem_type: str
    dataset_base_path: str
    dataset_id: int
    response_modes: Sequence[ResponseMode]
    train_set_size: int
    test_set_size: int
    batch_size: int
    learning_rate: float
    max_new_tokens: int
    output_dir_path_prefix: Optional[str]
    use_in_context_learning: bool
    n_in_context_learning_examples: int


def _load_classifier_problem(config: ExperimentConfig) -> tuple[Dataset, Problem]:
    classifier_class: Type[Classifier]
    problem_class: Type[ClassifierProblem[Any]]
    if config.problem_type == "decision_tree":
        classifier_class = DecisionTreeClassifier
        problem_class = DecisionTreeProblem
    elif config.problem_type == "logistic_regressor":
        classifier_class = LogisticRegressor
        problem_class = LogisticRegressorProblem
    else:
        raise ValueError(f"Unsupported problem type {config.problem_type}")
    dataset = load_from_disk(f"{config.dataset_base_path}/dataset_{config.dataset_id}")
    with open(
        os.path.join(config.dataset_base_path, f"classifier_{config.dataset_id}.json")
    ) as f:
        data = json.load(f)
    classifier = classifier_class.from_dict(data)
    problem = problem_class(classifier)
    if config.use_in_context_learning:
        kwargs = classifier_class.get_kwargs_from_example_instance(classifier)
        dataset_generator = DatasetGenerator(classifier_class, problem_class, kwargs)
        return dataset, FinetunedICLProblem(
            dataset_generator=dataset_generator,
            test_problem=problem,
            n_examples=config.n_in_context_learning_examples,
        )
    return dataset, problem


def load_problem(config: ExperimentConfig) -> tuple[Dataset, Problem]:
    if config.problem_type in ("decision_tree", "logistic_regressor"):
        return _load_classifier_problem(config)
    elif config.problem_type == "nlp_decision_tree":
        dataset = load_from_disk(f"{config.dataset_base_path}")
        if config.use_in_context_learning:
            raise NotImplementedError
        return dataset, NLPDecisionTreeProblem()
    else:
        raise ValueError(f"Invalid problem type: {config.problem_type}")


def print_dataset(
    dataset: Dataset, n_samples: Optional[int] = None, file: TextIO = sys.stdout
) -> None:
    if n_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
    for row in dataset:
        print("{", file=file)
        for key, value in row.items():
            print(f'    "{key}": {json.dumps(value)},', file=file)
        print("}", file=file)


def run(config: ExperimentConfig) -> None:
    llm, tokenizer = load_pretrained_llm(config.model_name)
    llm = convert_llm_to_peft(llm)
    dataset, problem = load_problem(config)

    llm_config = get_llm_config(llm)
    dataset["train"] = (
        dataset["train"]
        .select(range(config.train_set_size))
        .map(
            lambda x: build_finetuning_dataset(
                x,
                DatasetType.TRAIN,
                problem=problem,
                response_modes=config.response_modes,
            )
        )
    )
    dataset["test"] = (
        dataset["test"]
        .select(range(config.test_set_size))
        .map(
            lambda x: build_finetuning_dataset(
                x,
                DatasetType.TEST,
                problem=problem,
                response_modes=config.response_modes,
            )
        )
    )
    training_dataset = dataset.map(
        formatting_prompts_func,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    output_dir_path = None
    if config.output_dir_path_prefix is not None:
        output_dir_path = f"{config.output_dir_path_prefix}_{config.dataset_base_path.replace('/', '_')}_id_{config.dataset_id}_{'_'.join([response_mode.name for response_mode in config.response_modes])}{'_ICL' + str(config.n_in_context_learning_examples) if config.use_in_context_learning else ''}_{config.model_name.replace('/', '_')}_train_{config.train_set_size}_test_{config.test_set_size}_{str(datetime.datetime.now()).replace(' ', '_')}"
        os.makedirs(output_dir_path, exist_ok=False)
    if not isinstance(llm, DummyLLM):
        log_path = (
            os.path.join(output_dir_path, "training_log.jsonl")
            if output_dir_path is not None
            else None
        )
        train_llm(
            llm,
            tokenizer,
            training_dataset,
            log_path=log_path,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )
        FastLanguageModel.for_inference(llm)
    inference_test_dataset = dataset["test"].map(
        add_batch_inference,
        fn_kwargs={
            "llm": llm,
            "tokenizer": tokenizer,
            "llm_config": llm_config,
            "max_new_tokens": config.max_new_tokens,
        },
        batched=True,
        batch_size=8,
    )
    parsed_test_dataset = inference_test_dataset.map(
        parse_outputs,
        fn_kwargs={
            "problem": problem,
        },
    )

    print("\nSamples from the parsed test dataset:")
    print_dataset(parsed_test_dataset, n_samples=10)

    print("\nMetrics:")
    metrics = calculate_metrics(parsed_test_dataset, show=True)

    if output_dir_path is not None:
        inference_test_dataset.save_to_disk(
            os.path.join(output_dir_path, "inference_test_dataset")
        )
        with open(
            os.path.join(output_dir_path, "inference_test_dataset.txt"), "w"
        ) as f:
            print_dataset(inference_test_dataset, file=f)
        parsed_test_dataset.save_to_disk(
            os.path.join(output_dir_path, "parsed_test_dataset")
        )
        with open(os.path.join(output_dir_path, "parsed_test_dataset.txt"), "w") as f:
            print_dataset(parsed_test_dataset, file=f)
        with open(os.path.join(output_dir_path, "experiment_config.json"), "w") as f:
            config_dict = dataclasses.asdict(config)
            json.dump(config_dict, f, ensure_ascii=False, default=lambda obj: obj.name)
        with open(os.path.join(output_dir_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, ensure_ascii=False, ignore_nan=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--problem-type",
        type=str,
        required=True,
        choices=["decision_tree", "logistic_regressor", "nlp_decision_tree"],
    )
    parser.add_argument("--dataset-base-path", type=str, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument(
        "--response-modes",
        type=str,
        required=True,
        nargs="+",
        choices=["decision", "explanation", "reasoning"],
    )
    parser.add_argument("--train-set-size", type=int, required=True)
    parser.add_argument("--test-set-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir-path-prefix", type=str, required=False)
    parser.add_argument("--in-context-learning", action="store_true")
    parser.add_argument("--in-context-learning-examples", type=int)
    args = parser.parse_args()
    config = ExperimentConfig(
        model_name=args.model_name,
        problem_type=args.problem_type,
        dataset_base_path=args.dataset_base_path,
        dataset_id=args.dataset_id,
        response_modes=[ResponseMode(x.upper()) for x in args.response_modes],
        train_set_size=args.train_set_size,
        test_set_size=args.test_set_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        output_dir_path_prefix=args.output_dir_path_prefix,
        use_in_context_learning=args.in_context_learning,
        n_in_context_learning_examples=args.in_context_learning_examples,
    )
    run(config)
