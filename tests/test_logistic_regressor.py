import numpy as np

from explainable_llms.model.logistic_regressor import LogisticRegressor


def test_basic_functionality() -> None:
    classifier = LogisticRegressor(n_inputs=5)
    X = np.random.rand(20, 5)
    Y_pred = classifier.predict(X)
    for x, y_pred in zip(X, Y_pred):
        classifier.get_explanation_json(x)
