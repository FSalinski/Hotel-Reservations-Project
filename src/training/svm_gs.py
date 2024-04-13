"""
Script for training and optimizing SVM model using GridSearchCV
We'll optimize the model for highest possible recall
"""

import logging
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, KFold

import warnings

warnings.filterwarnings("ignore")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    df_train = pd.read_csv("data/final_train_train.csv")
    df_val = pd.read_csv("data/final_train_val.csv")
    logging.info("Data loaded")

    X_train = df_train.drop(columns=["booking_status"])
    y_train = df_train["booking_status"]
    X_val = df_val.drop(columns=["booking_status"])
    y_val = df_val["booking_status"]

    param_grid = {
        "C": [1, 2, 3, 4, 6],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "sigmoid"],
        "probability": [True],
        "random_state": [1],
    }

    grid = GridSearchCV(
        SVC(),
        param_grid,
        verbose=3,
        scoring="recall",
        cv=KFold(5, shuffle=True, random_state=1),
    )
    logging.info("GridSearchCV instance created")

    grid.fit(X_train, y_train)
    logging.info("GridSearchCV fitted")
    logging.info("Best parameters found: %s", grid.best_params_)

    model = grid.best_estimator_
    logging.info("Best model extracted")

    model.fit(X_train, y_train)
    logging.info("Model fitted")

    y_pred = model.predict(X_val)
    logging.info("Predictions made")

    logging.info("Confusion matrix:\n%s", confusion_matrix(y_val, y_pred))
    logging.info("Classification report:\n%s", classification_report(y_val, y_pred))

    with open("src/models/svc_gs.pkl", "wb") as f:
        pickle.dump(model, f)
    logging.info("Model saved")


if __name__ == "__main__":
    main()
