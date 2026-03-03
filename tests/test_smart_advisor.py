import pandas as pd
from pyml.intelligence.smart_advisor import SmartAdvisor


def test_smart_advisor_analysis():

    data = pd.DataFrame({
        "age": [22, 25, 30, 35],
        "salary": [20000, 25000, 30000, 35000],
        "label": [0, 1, 0, 1]
    })

    advisor = SmartAdvisor()
    report = advisor.analyze_data(data, target="label")

    assert "problem_type" in report
    assert report["problem_type"] == "Classification"