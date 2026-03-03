import pandas as pd
from pyml.visualization.advanced_plots import AdvancedVisualization


def test_visualization_basic():

    data = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1]
    })

    viz = AdvancedVisualization()

    # Only check if function runs without error
    viz.histogram(data, "a")