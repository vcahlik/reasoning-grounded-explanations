from datasets import Dataset
import numpy as np
import json
import math
from typing import Sequence


def calculate_failure(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("Input arrays must have equal dimensions.")
    n_valid = np.sum(((a == 0) | (a == 1)) & ((b == 0) | (b == 1)))
    n_total = len(a)
    if n_total == 0:
        return float(np.nan)
    return float(round(1 - (n_valid / n_total), 4))


def calculate_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Input arrays must have equal dimensions.")
    n_valid_and_equal = np.sum((a == b) & ((a == 0) | (a == 1)))
    n_valid = np.sum(((a == 0) | (a == 1)) & ((b == 0) | (b == 1)))
    if n_valid == 0:
        return float(np.nan)
    return float(round(1 - (n_valid_and_equal / n_valid), 4))


def calculate_failure_or_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Input arrays must have equal dimensions.")
    n_valid_and_equal = np.sum((a == b) & ((a == 0) | (a == 1)))
    n_total = len(a)
    if n_total == 0:
        return float(np.nan)
    return float(round(1 - (n_valid_and_equal / n_total), 4))


def calculate_json_metrics(
    target_jsons: Sequence[str], output_jsons: Sequence[str]
) -> tuple[float, float]:
    n_instances = len(target_jsons)
    n_failures = 0
    n_errors = 0
    for target_json, output_json in zip(target_jsons, output_jsons):
        target_data = json.loads(target_json)
        try:
            output_data = json.loads(output_json)
        except Exception:
            n_failures += 1
            continue
        if target_data != output_data:
            n_errors += 1
    failure_rate = n_failures / n_instances
    if n_instances == n_failures:
        error_rate = np.nan
    else:
        error_rate = n_errors / (n_instances - n_failures)
    return failure_rate, error_rate


def error_rate_to_accuracy(error_rate: float) -> float:
    if math.isnan(error_rate):
        return error_rate
    return float(round(1 - error_rate, 4))


def calculate_metrics(
    parsed_test_dataset: Dataset, show: bool = False
) -> dict[str, float]:
    metrics = {}
    if (
        "decision_output" in parsed_test_dataset.column_names
        or "explanation_output" in parsed_test_dataset.column_names
    ):
        decision_targets = np.array(parsed_test_dataset["decision_target"])
    if "decision_output" in parsed_test_dataset.column_names:
        decision_outputs = np.array(parsed_test_dataset["decision_output"])
        metrics["decision_failure"] = calculate_failure(
            decision_targets, decision_outputs
        )
        decision_failure_or_error = calculate_failure_or_error(
            decision_targets, decision_outputs
        )
        decision_error = calculate_error(decision_targets, decision_outputs)
        metrics["decision_failure_or_error"] = decision_failure_or_error
        metrics["decision_error"] = decision_error
        metrics["decision_accuracy_all"] = error_rate_to_accuracy(
            decision_failure_or_error
        )
        metrics["decision_accuracy_valid"] = error_rate_to_accuracy(decision_error)
    if "explanation_output" in parsed_test_dataset.column_names:
        explanation_decision_outputs = np.array(
            parsed_test_dataset["explanation_decision_output"]
        )
        metrics["explanation_decision_failure"] = calculate_failure(
            decision_targets, explanation_decision_outputs
        )
        explanation_decision_failure_or_error = calculate_failure_or_error(
            decision_targets, explanation_decision_outputs
        )
        explanation_decision_error = calculate_error(
            decision_targets, explanation_decision_outputs
        )
        metrics["explanation_decision_failure_or_error"] = (
            explanation_decision_failure_or_error
        )
        metrics["explanation_decision_error"] = explanation_decision_error
        metrics["explanation_decision_accuracy_all"] = error_rate_to_accuracy(
            explanation_decision_failure_or_error
        )
        metrics["explanation_decision_accuracy_valid"] = error_rate_to_accuracy(
            explanation_decision_error
        )
    if (
        "decision_output" in parsed_test_dataset.column_names
        and "explanation_output" in parsed_test_dataset.column_names
    ):
        metrics["decision_explanation_alignment_failure"] = calculate_failure(
            decision_outputs, explanation_decision_outputs
        )
        decision_explanation_alignment_failure_or_error = calculate_failure_or_error(
            decision_outputs, explanation_decision_outputs
        )
        decision_explanation_alignment_error = calculate_error(
            decision_outputs, explanation_decision_outputs
        )
        metrics["decision_explanation_alignment_failure_or_error"] = (
            decision_explanation_alignment_failure_or_error
        )
        metrics["decision_explanation_alignment_error"] = (
            decision_explanation_alignment_error
        )
        metrics["decision_explanation_alignment_accuracy_all"] = error_rate_to_accuracy(
            decision_explanation_alignment_failure_or_error
        )
        metrics["decision_explanation_alignment_accuracy_valid"] = (
            error_rate_to_accuracy(decision_explanation_alignment_error)
        )
    if "reasoning_output" in parsed_test_dataset.column_names:
        reasoning_failure, reasoning_error = calculate_json_metrics(
            parsed_test_dataset["reasoning_target"],
            parsed_test_dataset["reasoning_output"],
        )
        metrics["reasoning_failure"] = reasoning_failure
        metrics["reasoning_error"] = reasoning_error
    if show:
        print(json.dumps(metrics, indent=4))
    return metrics
