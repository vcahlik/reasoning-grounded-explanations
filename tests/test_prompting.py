from explainable_llms.training import (
    build_prompting_dataset,
    add_batch_inference,
    parse_outputs,
    ResponseMode,
    build_finetuning_dataset,
)
from explainable_llms.evaluation import calculate_metrics
from explainable_llms.llm import LLM, get_llm_config
from explainable_llms.problem import (
    Problem,
    DecisionTreeProblem,
    NLPDecisionTreeProblem,
    LogisticRegressorProblem,
)
from explainable_llms.model.decision_tree import DecisionTreeClassifier
from explainable_llms.utils import DatasetType
from explainable_llms.model.logistic_regressor import LogisticRegressor
from transformers import PreTrainedTokenizer
from .utils import (
    DECISION_TREE_DATASET_PATH,
    LOGISTIC_REGRESSOR_DATASET_PATH,
    NLP_DECISION_TREE_DATASET_PATH,
)

from datasets import load_from_disk, disable_caching, Dataset
from unsloth import FastLanguageModel
import json
from typing import Optional


disable_caching()


def _test_prompting(
    dataset: Dataset,
    problem: Problem,
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    use_problem_structure: bool,
    n_example_xs: Optional[int] = 3,
) -> None:
    llm_config = get_llm_config(llm)
    dataset["train"] = dataset["train"].select(range(10))
    dataset["test"] = dataset["test"].select(range(2))
    dataset = dataset.map(
        build_prompting_dataset,
        fn_kwargs={
            "problem": problem,
            "use_problem_structure": use_problem_structure,
            "response_modes": [ResponseMode.DECISION, ResponseMode.EXPLANATION],
            "n_example_xs": n_example_xs,
            "reveal_classifier_type": True,
        },
    )
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


def _test_decision_tree_prompting(
    llm: LLM, tokenizer: PreTrainedTokenizer, use_problem_structure: bool
) -> None:
    base_dataset_dir = DECISION_TREE_DATASET_PATH
    dataset = load_from_disk(f"{base_dataset_dir}/dataset_0")
    with open(f"{base_dataset_dir}/classifier_0.json") as f:
        data = json.load(f)
    tree = DecisionTreeClassifier.from_dict(data)
    problem = DecisionTreeProblem(tree)
    _test_prompting(
        dataset, problem, llm, tokenizer, use_problem_structure=use_problem_structure
    )


def _test_logistic_regressor_prompting(
    llm: LLM, tokenizer: PreTrainedTokenizer, use_problem_structure: bool
) -> None:
    base_dataset_dir = LOGISTIC_REGRESSOR_DATASET_PATH
    dataset = load_from_disk(f"{base_dataset_dir}/dataset_0")
    with open(f"{base_dataset_dir}/classifier_0.json") as f:
        data = json.load(f)
    classifier = LogisticRegressor.from_dict(data)
    problem = LogisticRegressorProblem(classifier)
    _test_prompting(
        dataset, problem, llm, tokenizer, use_problem_structure=use_problem_structure
    )


def _test_nlp_decision_tree_prompting(
    llm: LLM, tokenizer: PreTrainedTokenizer, use_problem_structure: bool
) -> None:
    dataset = load_from_disk(NLP_DECISION_TREE_DATASET_PATH)
    problem = NLPDecisionTreeProblem()
    split_dataset = dataset["train"].train_test_split(train_size=100, shuffle=True)
    dataset = split_dataset["train"].train_test_split(train_size=50, shuffle=True)
    example_dataset = (
        split_dataset["test"]
        .select(range(2))
        .map(
            lambda x: build_finetuning_dataset(
                x,
                dataset_type=DatasetType.TEST,
                problem=problem,
            )
        )
    )
    problem.example_dataset = example_dataset
    _test_prompting(
        dataset,
        problem,
        llm,
        tokenizer,
        use_problem_structure=use_problem_structure,
        n_example_xs=None,
    )


def test_decision_tree_prompting_with_few_shot_prompt(
    llm: LLM, tokenizer: PreTrainedTokenizer
) -> None:
    _test_decision_tree_prompting(llm, tokenizer, use_problem_structure=False)


def test_decision_tree_prompting_with_tree_structure_prompt(
    llm: LLM, tokenizer: PreTrainedTokenizer
) -> None:
    _test_decision_tree_prompting(llm, tokenizer, use_problem_structure=True)


def test_logistic_regressor_prompting_with_few_shot_prompt(
    llm: LLM, tokenizer: PreTrainedTokenizer
) -> None:
    _test_logistic_regressor_prompting(llm, tokenizer, use_problem_structure=False)


def test_nlp_decision_tree_prompting_with_few_shot_prompt(
    llm: LLM, tokenizer: PreTrainedTokenizer
) -> None:
    _test_nlp_decision_tree_prompting(llm, tokenizer, use_problem_structure=False)
