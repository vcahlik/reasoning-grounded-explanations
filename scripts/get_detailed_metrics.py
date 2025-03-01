import numpy as np
from datasets import load_from_disk
import argparse
import ast
import math

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

def parse_reasoning_string(s: str) -> list:
    return [int(x) for x in s.split(',')]

def parse_explanation_string(s: str) -> list:
    # Replace lowercase true/false with Python's True/False
    s = s.replace('true', 'True').replace('false', 'False')
    data = ast.literal_eval(s)
    results = []
    for item in data:
        if item[1] == True:
            results.append(1)
        elif item[1] == False:
            results.append(0)
        else:
            results.append(int(item[1]))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)

    # 1) Decision target vs output accuracy
    decision_targets = np.array([int(x['decision_target']) for x in dataset])
    decision_outputs = np.array([int(x['decision_output']) for x in dataset])
    accuracy_1 = error_rate_to_accuracy(calculate_failure_or_error(decision_targets, decision_outputs))
    print(f"1. Decision target-output accuracy: {accuracy_1}")

    # 2) Decision output vs last reasoning output
    reasoning_outputs = np.array([parse_reasoning_string(x['reasoning_output'])[-1] for x in dataset])
    accuracy_2 = error_rate_to_accuracy(calculate_failure_or_error(decision_outputs, reasoning_outputs))
    print(f"2. Decision output vs last reasoning accuracy: {accuracy_2}")

    # 3) Reasoning target vs output per position
    reasoning_targets = np.array([parse_reasoning_string(x['reasoning_target']) for x in dataset])
    reasoning_outputs_full = np.array([parse_reasoning_string(x['reasoning_output']) for x in dataset])
    
    print("3. Reasoning target-output accuracy per position:")
    for pos in range(8):
        pos_accuracy = error_rate_to_accuracy(
            calculate_failure_or_error(
                reasoning_targets[:, pos],
                reasoning_outputs_full[:, pos]
            )
        )
        print(f"   Position {pos}: {pos_accuracy}")

    # 4) Explanation output vs reasoning target per position
    explanation_outputs = np.array([parse_explanation_string(x['explanation_output']) for x in dataset])
    
    print("4. Explanation output vs reasoning target accuracy per position:")
    for pos in range(8):
        pos_accuracy = error_rate_to_accuracy(
            calculate_failure_or_error(
                reasoning_targets[:, pos],
                explanation_outputs[:, pos]
            )
        )
        print(f"   Position {pos}: {pos_accuracy}")

    # 5) Decision output vs last explanation output
    explanation_last = np.array([parse_explanation_string(x['explanation_output'])[-1] for x in dataset])
    accuracy_5 = error_rate_to_accuracy(calculate_failure_or_error(decision_outputs, explanation_last))
    print(f"5. Decision output vs last explanation accuracy: {accuracy_5}")

    # 6) Explanation outputs vs reasoning outputs per position
    print("6. Explanation output vs reasoning output accuracy per position:")
    for pos in range(8):
        pos_accuracy = error_rate_to_accuracy(
            calculate_failure_or_error(
                reasoning_outputs_full[:, pos],
                explanation_outputs[:, pos]
            )
        )
        print(f"   Position {pos}: {pos_accuracy}")

if __name__ == "__main__":
    main()
