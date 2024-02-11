"""
Script for training and optimizing SVM model using GridSearchCV
We'll optimize the model for highest possible recall
"""

import logging
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings("ignore")


def main():
    pass

if __name__ == "__main__":
    main()
