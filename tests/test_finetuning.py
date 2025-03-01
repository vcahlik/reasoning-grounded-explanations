from datasets import Dataset, load_from_disk, disable_caching
from unsloth import FastLanguageModel
import json
from transformers import PreTrainedTokenizer
from typing import Sequence, Optional
import os

from explainable_llms.training import (
    formatting_prompts_func,
    build_finetuning_dataset,
    train_llm,
    add_batch_inference,
    parse_outputs,
    ResponseMode,
)
from explainable_llms.evaluation import calculate_metrics
from explainable_llms.llm import LLM, get_llm_config, DummyLLM, convert_llm_to_peft
from explainable_llms.model.decision_tree import DecisionTreeClassifier
from explainable_llms.model.logistic_regressor import LogisticRegressor
from explainable_llms.dataset_generator import DatasetGenerator
from explainable_llms.utils import DatasetType
from explainable_llms.problem import (
    Problem,
    DecisionTreeProblem,
    LogisticRegressorProblem,
    NLPDecisionTreeProblem,
    FinetunedICLProblem,
)
from .utils import (
    DECISION_TREE_DATASET_PATH,
    LOGISTIC_REGRESSOR_DATASET_PATH,
    NLP_DECISION_TREE_DATASET_PATH,
)


disable_caching()


def _test_finetuning(
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    problem: Problem,
    response_modes: Optional[Sequence[ResponseMode]] = None,
) -> None:
    llm_config = get_llm_config(llm)
    dataset["train"] = dataset["train"].select(range(2))
    dataset["test"] = dataset["test"].select(range(2))
    dataset["train"] = dataset["train"].map(
        build_finetuning_dataset,
        fn_kwargs={
            "dataset_type": DatasetType.TRAIN,
            "problem": problem,
            "response_modes": response_modes,
        },
    )
    dataset["test"] = dataset["test"].map(
        build_finetuning_dataset,
        fn_kwargs={
            "dataset_type": DatasetType.TEST,
            "problem": problem,
            "response_modes": response_modes,
        },
    )
    training_dataset = dataset.map(
        formatting_prompts_func,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    if not isinstance(llm, DummyLLM):
        train_llm(llm, tokenizer, training_dataset, max_steps=1)
        FastLanguageModel.for_inference(llm)
    inference_test_dataset = dataset["test"].map(
        add_batch_inference,
        fn_kwargs={
            "llm": llm,
            "tokenizer": tokenizer,
            "llm_config": llm_config,
        },
        batched=True,
        batch_size=2,
    )
    parsed_test_dataset = inference_test_dataset.map(
        parse_outputs,
        fn_kwargs={
            "problem": problem,
        },
    )
    calculate_metrics(parsed_test_dataset, show=True)


def _test_decision_tree_finetuning(
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    response_modes: Optional[Sequence[ResponseMode]] = None,
    in_context_learning: bool = False,
) -> None:
    dataset = load_from_disk(os.path.join(DECISION_TREE_DATASET_PATH, "dataset_0"))
    with open(os.path.join(DECISION_TREE_DATASET_PATH, "classifier_0.json")) as f:
        data = json.load(f)
    classifier = DecisionTreeClassifier.from_dict(data)
    problem = DecisionTreeProblem(classifier)
    if in_context_learning:
        kwargs = DecisionTreeClassifier.get_kwargs_from_example_instance(classifier)
        dataset_generator = DatasetGenerator(
            DecisionTreeClassifier, DecisionTreeProblem, kwargs
        )
        problem = FinetunedICLProblem(
            dataset_generator=dataset_generator, test_problem=problem, n_examples=2
        )  # type: ignore
    _test_finetuning(llm, tokenizer, dataset, problem, response_modes)


def _test_logistic_regressor_finetuning(
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    response_modes: Optional[Sequence[ResponseMode]] = None,
    in_context_learning: bool = False,
) -> None:
    dataset = load_from_disk(os.path.join(LOGISTIC_REGRESSOR_DATASET_PATH, "dataset_0"))
    with open(os.path.join(LOGISTIC_REGRESSOR_DATASET_PATH, "classifier_0.json")) as f:
        data = json.load(f)
    classifier = LogisticRegressor.from_dict(data)
    problem = LogisticRegressorProblem(classifier)
    if in_context_learning:
        kwargs = LogisticRegressor.get_kwargs_from_example_instance(classifier)
        dataset_generator = DatasetGenerator(
            LogisticRegressor, LogisticRegressorProblem, kwargs
        )
        problem = FinetunedICLProblem(
            dataset_generator=dataset_generator, test_problem=problem, n_examples=2
        )  # type: ignore
    _test_finetuning(llm, tokenizer, dataset, problem, response_modes)


def _test_nlp_decision_tree_finetuning(
    tokenizer: PreTrainedTokenizer,
) -> None:
    llm = DummyLLM()
    dataset = load_from_disk(NLP_DECISION_TREE_DATASET_PATH)
    problem = NLPDecisionTreeProblem()
    _test_finetuning(llm, tokenizer, dataset, problem)


def test_decision_tree_decision_finetuning_with_real_llm(
    llm: LLM, tokenizer: PreTrainedTokenizer
) -> None:
    llm = convert_llm_to_peft(llm)
    _test_decision_tree_finetuning(
        llm, tokenizer, response_modes=[ResponseMode.DECISION]
    )


def test_decision_tree_explanation_finetuning(tokenizer: PreTrainedTokenizer) -> None:
    llm = DummyLLM()
    _test_decision_tree_finetuning(
        llm, tokenizer, response_modes=[ResponseMode.EXPLANATION]
    )


def test_decision_tree_decision_explanation_finetuning(
    tokenizer: PreTrainedTokenizer,
) -> None:
    llm = DummyLLM()
    _test_decision_tree_finetuning(
        llm,
        tokenizer,
        response_modes=[ResponseMode.DECISION, ResponseMode.EXPLANATION],
    )


def test_decision_tree_finetuning_icl(
    tokenizer: PreTrainedTokenizer,
) -> None:
    llm = DummyLLM()
    _test_decision_tree_finetuning(llm, tokenizer, in_context_learning=True)


def test_decision_tree_finetuning_with_reasoning(
    tokenizer: PreTrainedTokenizer,
) -> None:
    llm = DummyLLM()
    _test_decision_tree_finetuning(llm, tokenizer)


def test_logistic_regressor_finetuning_with_reasoning(
    tokenizer: PreTrainedTokenizer,
) -> None:
    llm = DummyLLM()
    _test_logistic_regressor_finetuning(llm, tokenizer)


def test_logistic_regressor_finetuning_icl(
    tokenizer: PreTrainedTokenizer,
) -> None:
    llm = DummyLLM()
    _test_logistic_regressor_finetuning(llm, tokenizer, in_context_learning=True)


def test_nlp_decision_tree_finetuning(tokenizer: PreTrainedTokenizer) -> None:
    _test_nlp_decision_tree_finetuning(tokenizer)
