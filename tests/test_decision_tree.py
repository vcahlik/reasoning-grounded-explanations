import sys
from io import StringIO
import numpy as np

from explainable_llms.model.decision_tree import DecisionTreeClassifier


def test_basic_functionality() -> None:
    tree = DecisionTreeClassifier(n_inputs=2, depth=8)
    X = np.random.rand(20, 2)
    Y_pred = tree.predict(X)
    for x, y_pred in zip(X, Y_pred):
        pseudocode = tree.get_pseudocode(tuple(x))

        original_stdout = sys.stdout
        exec_output = sys.stdout = StringIO()
        exec(pseudocode)
        sys.stdout = original_stdout
        y_pred_pseudocode = int(exec_output.getvalue())

        assert y_pred == y_pred_pseudocode

        tree.get_explanation_json(x)

    tree.plot()
    tree_dict = tree.to_dict()
    loaded_tree = DecisionTreeClassifier.from_dict(tree_dict)
    assert np.array_equal(loaded_tree.predict(X), Y_pred)
