import os
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set the style parameters
plt.rcParams.update(
    {
        "legend.fontsize": 18,
        "font.family": "serif",
    }
)

base_dir = "outputs/decision_tree_depths"


def extract_depth(folder_name):
    match = re.search(r"depth_(\d+)", folder_name)
    return int(match.group(1)) if match else None


# Prepare data for plotting
data = []
for folder_name in os.listdir(base_dir):
    if "decision_trees_v10" not in folder_name:
        continue

    depth = extract_depth(folder_name)
    if not depth:
        continue

    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Determine if reasoning is used
    reasoning = "REASONING" in folder_name

    metrics_file = os.path.join(folder_path, "metrics.json")
    if not os.path.exists(metrics_file):
        continue

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Get decision and explanation accuracies
    decision_acc = metrics.get("decision_accuracy_all")
    explanation_acc = metrics.get("explanation_decision_accuracy_all")

    data.append(
        {
            "depth": depth,
            "accuracy": decision_acc,
            "type": "Decisions" + (" (with reasoning)" if reasoning else ""),
        }
    )
    data.append(
        {
            "depth": depth,
            "accuracy": explanation_acc,
            "type": "Explanations" + (" (with reasoning)" if reasoning else ""),
        }
    )

# Convert to DataFrame and sort by depth
df = pd.DataFrame(data)
df = df.sort_values("depth")
df["depth"] = df["depth"].apply(lambda x: x + 1)

# Create the plot with specific styling
fig, ax = plt.subplots(figsize=(12, 6))

# Define style for each line type
styles = {
    "Decisions": {"linestyle": "--", "marker": "s", "alpha": 0.6},
    "Explanations": {"linestyle": "-", "marker": "s", "alpha": 0.6},
    "Decisions (with reasoning)": {"linestyle": "--", "marker": "o", "alpha": 0.6},
    "Explanations (with reasoning)": {"linestyle": "-", "marker": "o", "alpha": 0.6},
}

# Plot each line separately to control styles
for line_type in styles:
    mask = df["type"] == line_type
    sns.lineplot(
        data=df[mask],
        x="depth",
        y="accuracy",
        label=line_type,
        marker=styles[line_type]["marker"],
        linestyle=styles[line_type]["linestyle"],
        alpha=styles[line_type]["alpha"],
        markersize=14,
        linewidth=2.5,
    )

# Customize the plot
# ax.set_title("Decision Tree Performance vs Depth", fontsize=20, pad=20)
ax.set_xlabel("Tree Depth", fontsize=16, labelpad=10)
ax.set_ylabel("Classification Accuracy", fontsize=16, labelpad=10)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.grid(True, linestyle="--", alpha=0.7)

# Customize legend
legend = ax.get_legend()
# legend.set_title("Metric Type", prop={"size": 18})
legend.set_frame_on(True)
for text in legend.get_texts():
    text.set_fontsize(16)

# Set y-axis limits with some padding
ax.set_ylim(
    np.floor(df["accuracy"].min() * 10) / 10 - 0.05,
    np.ceil(df["accuracy"].max() * 10) / 10 + 0.05,
)

# Adjust layout and save
plt.tight_layout()
output_file = "depth_performance.pdf"
if os.path.exists(output_file):
    os.remove(output_file)
plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
plt.close()
