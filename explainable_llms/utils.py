import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from typing import Optional, Sequence, Any
from matplotlib.gridspec import GridSpec
from enum import Enum

from explainable_llms.model.decision_tree import DecisionTreeClassifier


plt.rcParams.update(
    {
        "legend.fontsize": 18,
        "font.family": "serif",
    }
)


class DatasetType(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"


def plot_errors(
    dataset: Dataset,
    tree: Optional[DecisionTreeClassifier] = None,
    show: bool = True,
    output_file_name: Optional[str] = None,
) -> None:
    def _plot_axis(
        ax: plt.Axes,
        targets: Sequence[dict[str, Any]],
        outputs: Sequence[dict[str, Any]],
        mode: str,
    ) -> None:
        assert mode in ("decision", "explanation")

        errors = np.array(targets) != np.array(outputs)
        if tree is not None:
            tree.plot(show=False, ax=ax)
        sns.scatterplot(
            ax=ax,
            x=np.array(X)[errors],
            y=np.array(Y)[errors],
            color="firebrick",
            label="Error",
            marker="X",
            s=500,
        )
        ax = sns.scatterplot(
            ax=ax, x=X, y=Y, hue=outputs, s=80, edgecolor="w", linewidth=2
        )
        legend = ax.get_legend()
        legend.set_title(legend.get_title().get_text(), prop={"size": 12})
        for text in legend.get_texts():
            original_label = text.get_text()
            text.set_text(legend_renaming_map.get(original_label, original_label))
        sns.move_legend(ax, "upper right")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", which="major", labelsize=14)
        if mode == "decision":
            ax.set_xticklabels(
                [label.get_text() for label in ax0.get_xticklabels()[:-1]]
            )  # Remove the last tick label
            ax.set_title("Answers", fontsize=20)
        else:
            ax.set_yticks([])
            ax.set_title("Explanations", fontsize=20)

    if len(dataset) > 0 and len(dataset["x"][0]) != 2:
        raise NotImplementedError(
            "Errors can't be plotted as only 2 input variables are supported"
        )

    legend_renaming_map = {"0": "Pred. 0", "1": "Pred. 1"}
    X = [x[0] for x in dataset["x"]]
    Y = [x[1] for x in dataset["x"]]

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.025)  # Adjust wspace as needed
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    _plot_axis(
        ax=ax0,
        targets=dataset["decision_target"],
        outputs=dataset["decision_output"],
        mode="decision",
    )
    _plot_axis(
        ax=ax1,
        targets=dataset["decision_target"],
        outputs=dataset["explanation_decision_output"],
        mode="explanation",
    )

    if show:
        plt.show()
    if output_file_name is not None:
        fig.savefig(output_file_name, bbox_inches="tight")
