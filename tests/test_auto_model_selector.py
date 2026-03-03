import pandas as pd
from pyml.intelligence.auto_model_selector import AutoModelSelector


def test_auto_model_selector():

    data = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5],
        "x2": [5, 4, 3, 2, 1],
        "y": [0, 1, 0, 1, 0]
    })

    X = data[["x1", "x2"]]
    y = data["y"]

    selector = AutoModelSelector()
    best = selector.best_model(X, y)

    assert best is not None