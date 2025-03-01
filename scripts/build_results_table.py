import os
import json

base_dir = "outputs/complete_v1.5"

# Define table configurations
TABLE_CONFIGS = {
    "logistic": {
        "caption": "Experiments with Logistic Regression",
        "columns": ["Llama 3 8B", "Mistral NeMo", "Mistral 7B", "Zephyr SFT", "Phi-4"],
        "pattern": "logistic_regressors",
        "metric_type": "accuracy",
        "metrics": ["Answer acc.", "Explanation acc.", "Alignment rate"],
    },
    "decision_trees": {
        "caption": "Experiments with Decision Trees",
        "columns": ["Llama 3 8B", "Mistral NeMo", "Mistral 7B", "Zephyr SFT", "Phi-4"],
        "pattern": "decision_trees_v10",
        "metric_type": "accuracy",
        "metrics": ["Answer acc.", "Explanation acc.", "Alignment rate"],
    },
    "natural_trees": {
        "caption": "Experiments with Natural Language Decision Trees",
        "columns": ["Llama 3 8B", "Mistral NeMo", "Mistral 7B", "Zephyr SFT", "Phi-4"],
        "pattern": "mortgages_decision_tree",
        "metric_type": "accuracy",
        "metrics": ["Answer acc.", "Explanation acc.", "Alignment rate"],
    },
}

# Model name mapping
MODEL_MAPPING = {
    "llama-3-8b-Instruct": "Llama 3 8B",
    "Mistral-Nemo-Instruct": "Mistral NeMo",
    "mistral-7b-instruct": "Mistral 7B",
    "zephyr-sft": "Zephyr SFT",
    "phi-4": "Phi-4",
}


def initialize_data_structure(table_config):
    data = {
        "Ans./exp. training": [],
        "ICL": [],
        "Reasoning": [],
        "Metric": table_config["metrics"],
    }

    # Add columns for values and failure rates
    for col in table_config["columns"]:
        data[col] = []
        data[f"{col}_fail"] = []

    # Initialize with configurations - different for natural trees
    if table_config["pattern"] == "mortgages_decision_tree":
        configurations = [
            ("Separately", "No", "No"),
            ("Jointly", "No", "No"),
            ("Jointly", "No", "Yes"),
        ]
    else:
        configurations = [
            ("Separately", "Yes", "No"),
            ("Separately", "No", "No"),
            ("Jointly", "No", "No"),
            ("Jointly", "No", "Yes"),
        ]

    for jointly, icl, reasoning in configurations:
        data["Ans./exp. training"].extend([jointly] + [None, None])
        data["ICL"].extend([icl] + [None, None])
        data["Reasoning"].extend([reasoning] + [None, None])
        for col in table_config["columns"]:
            data[col].extend([None] * 3)
            data[f"{col}_fail"].extend([None] * 3)

    return data


def extract_model_name(folder_name):
    for key in MODEL_MAPPING:
        if key in folder_name:
            return MODEL_MAPPING[key]
    return None


def get_row_index(jointly, icl, reasoning, pattern):
    if pattern == "mortgages_decision_tree":
        configurations = [
            ("Separately", "No", "No"),
            ("Jointly", "No", "No"),
            ("Jointly", "No", "Yes"),
        ]
    else:
        configurations = [
            ("Separately", "Yes", "No"),
            ("Separately", "No", "No"),
            ("Jointly", "No", "No"),
            ("Jointly", "No", "Yes"),
        ]

    for i, config in enumerate(configurations):
        if config == (jointly, icl, reasoning):
            return i * 3
    return None


def process_folders(table_config):
    data = initialize_data_structure(table_config)

    for folder_name in os.listdir(base_dir):
        if table_config["pattern"] not in folder_name:
            continue

        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Determine experiment type
        icl = "Yes" if "ICL" in folder_name else "No"
        reasoning = "Yes" if "REASONING" in folder_name else "No"
        jointly = (
            "Separately"
            if not any(
                x in folder_name
                for x in ["DECISION_EXPLANATION_REASONING", "DECISION_EXPLANATION"]
            )
            else "Jointly"
        )

        # Extract model name for all tables
        column = extract_model_name(folder_name)
        if not column:
            continue

        try:
            metrics_file = os.path.join(folder_path, "metrics.json")
            if not os.path.exists(metrics_file):
                continue

            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            row_index = get_row_index(jointly, icl, reasoning, table_config["pattern"])
            if row_index is None:
                continue

            # Update metrics based on experiment type
            update_accuracy_metrics(data, metrics, column, row_index, folder_name)

        except Exception as e:
            print(f"Error processing {folder_name}: {e}")

    return data


def generate_latex_table(data, table_config):
    header = (
        r"\begin{table*}"
        + "\n"
        + rf"\caption{{{table_config['caption']}}}"
        + "\n"
        + r"\centering"
        + "\n"
        + r"\makebox[\textwidth][c]{"
        + "\n"
        + r"\begin{tabular}"
        + rf"{{llll|{len(table_config['columns'])*'c'}}}"
        + "\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Ans./exp. training} & \textbf{ICL} & \textbf{Reasoning} & \textbf{Metric} & "
        + " & ".join(rf"\textbf{{{col}}}" for col in table_config["columns"])
        + r" \\"
        + "\n"
        + r"\midrule"
    )

    body = ""
    for i in range(0, len(data["Ans./exp. training"]), 3):
        if data["Ans./exp. training"][i] is None:
            continue

        body += rf"\multirow{{3}}{{*}}{{{data['Ans./exp. training'][i]}}} & \multirow{{3}}{{*}}{{{data['ICL'][i]}}} & \multirow{{3}}{{*}}{{{data['Reasoning'][i]}}} & "

        for j in range(3):
            if j > 0:
                body += "& & & "
            body += f"{data['Metric'][j]} & "

            values = []
            for col in table_config["columns"]:
                val = data[col][i + j]
                fail_val = data.get(f"{col}_fail", [None])[i + j]

                if isinstance(val, float):
                    if isinstance(fail_val, float):
                        values.append(f"{val:.3f} ({fail_val:.3f})")
                    else:
                        values.append(f"{val:.3f}")
                else:
                    values.append("?")

            body += " & ".join(values) + r" \\"
            if j < 2:
                body += "\n"
            else:
                body += r"\midrule" + "\n"

    footer = (
        r"\bottomrule"
        + "\n"
        + r"\end{tabular}"
        + "\n"
        + r"}"
        + "\n"
        + r"\end{table*}"
        + "\n\n"
    )

    return header + "\n" + body + footer


def update_accuracy_metrics(data, metrics, column, row_index, folder_name):
    if (
        "DECISION_EXPLANATION_REASONING" in folder_name
        or "DECISION_EXPLANATION" in folder_name
    ):
        data[column][row_index] = metrics.get("decision_accuracy_all")
        data[f"{column}_fail"][row_index] = metrics.get("decision_failure")
        data[column][row_index + 1] = metrics.get("explanation_decision_accuracy_all")
        data[f"{column}_fail"][row_index + 1] = metrics.get(
            "explanation_decision_failure"
        )
        data[column][row_index + 2] = metrics.get(
            "decision_explanation_alignment_accuracy_all"
        )
        data[f"{column}_fail"][row_index + 2] = metrics.get(
            "decision_explanation_alignment_failure"
        )
    elif "DECISION_ICL" in folder_name or "DECISION" in folder_name:
        data[column][row_index] = metrics.get("decision_accuracy_all")
        data[f"{column}_fail"][row_index] = metrics.get("decision_failure")
    elif "EXPLANATION_ICL" in folder_name or "EXPLANATION" in folder_name:
        data[column][row_index + 1] = metrics.get("explanation_decision_accuracy_all")
        data[f"{column}_fail"][row_index + 1] = metrics.get(
            "explanation_decision_failure"
        )
        # Add alignment rate for separate training
        data[column][row_index + 2] = "?"


# Main execution
for table_name, config in TABLE_CONFIGS.items():
    data = process_folders(config)
    latex_table = generate_latex_table(data, config)
    print(latex_table)
