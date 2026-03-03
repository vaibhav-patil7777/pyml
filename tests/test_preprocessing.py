import pandas as pd
from pyml.preprocessing.scaling import Scaler


def test_scaler():

    data = pd.DataFrame({
        "x": [1, 2, 3, 4]
    })

    scaler = Scaler()
    scaled = scaler.standard_scale(data)

    assert scaled.shape == data.shape