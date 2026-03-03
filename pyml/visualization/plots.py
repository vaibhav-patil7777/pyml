"""
PyML Advanced & Interactive Visualization Module
Author: Vaibhav Arun Patil
Version: 0.3.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from ..core.base import BaseModule
from ..logger import logger


class AdvancedVisualization(BaseModule):

    def __init__(self):
        super().__init__()
        logger.info("Advanced Visualization Engine Initialized")

    # ================= BASIC PLOTS =================

    def histogram(self, data, column):
        data = self.convert_to_dataframe(data)
        plt.figure()
        sns.histplot(data[column], kde=True)
        plt.title(f"Histogram - {column}")
        plt.show()

    def bar(self, data, column):
        data = self.convert_to_dataframe(data)
        plt.figure()
        sns.countplot(x=data[column])
        plt.title(f"Bar Plot - {column}")
        plt.show()

    def box(self, data, column):
        data = self.convert_to_dataframe(data)
        plt.figure()
        sns.boxplot(y=data[column])
        plt.title(f"Box Plot - {column}")
        plt.show()

    # ================= MULTIVARIATE PLOTS =================

    def scatter(self, data, x, y, color=None):
        data = self.convert_to_dataframe(data)
        plt.figure()
        sns.scatterplot(data=data, x=x, y=y, hue=color)
        plt.title(f"Scatter Plot - {x} vs {y}")
        plt.show()

    def pairplot(self, data):
        data = self.convert_to_dataframe(data)
        sns.pairplot(data)
        plt.show()

    def correlation_heatmap(self, data):
        data = self.convert_to_dataframe(data)
        plt.figure()
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    # ================= MODEL VISUALIZATION =================

    def confusion_matrix(self, cm):
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()

    def feature_importance(self, features, importance):
        plt.figure()
        plt.barh(features, importance)
        plt.title("Feature Importance")
        plt.show()

    # ================= INTERACTIVE PLOTS =================

    def interactive_scatter(self, data, x, y, color=None):
        data = self.convert_to_dataframe(data)
        fig = px.scatter(data, x=x, y=y, color=color,
                         title=f"Interactive Scatter - {x} vs {y}")
        fig.show()

    def interactive_line(self, data, x, y):
        data = self.convert_to_dataframe(data)
        fig = px.line(data, x=x, y=y,
                      title=f"Interactive Line - {x} vs {y}")
        fig.show()

    def interactive_bar(self, data, column):
        data = self.convert_to_dataframe(data)
        fig = px.bar(data[column].value_counts().reset_index(),
                     x="index", y=column,
                     title=f"Interactive Bar - {column}")
        fig.show()