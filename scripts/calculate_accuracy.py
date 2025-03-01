import numpy as np
import math
from datasets import load_from_disk
import argparse
import os

def calculate_failure_or_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Input arrays must have equal dimensions.")
    n_valid_and_equal = np.sum((a == b) & ((a == 0) | (a == 1)))
    n_total = len(a)
    if n_total == 0:
        return float(np.nan)
    return float(round(1 - (n_valid_and_equal / n_total), 4))

def error_rate_to_accuracy(error_rate: float) -> float:
    if math.isnan(error_rate):
        return error_rate
    return float(round(1 - error_rate, 4))

def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy between decision_output and explanation_decision_output')
    parser.add_argument('decision_dataset_path', type=str, help='Local path to the dataset containing decision_output')
    parser.add_argument('explanation_dataset_path', type=str, help='Local path to the dataset containing explanation_decision_output')
    args = parser.parse_args()

    # Load the datasets from disk
    decision_dataset = load_from_disk(os.path.join("outputs/complete_v1.5", args.decision_dataset_path, "parsed_test_dataset"))
    explanation_dataset = load_from_disk(os.path.join("outputs/complete_v1.5", args.explanation_dataset_path, "parsed_test_dataset"))
    
    # Convert to arrays for comparison
    decision_outputs = np.array([example['decision_output'] for example in decision_dataset])
    explanation_decision_outputs = np.array([example['explanation_decision_output'] for example in explanation_dataset])
    
    # Calculate accuracy
    error_rate = calculate_failure_or_error(decision_outputs, explanation_decision_outputs)
    accuracy = error_rate_to_accuracy(error_rate)
    
    print(f"Accuracy between decision_output and explanation_decision_output: {accuracy}")

if __name__ == "__main__":
    main()
