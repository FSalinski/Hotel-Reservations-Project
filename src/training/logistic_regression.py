"""
Script for training a logistic regression model (will be used as a baseline model)
"""
import logging
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import warnings

warnings.filterwarnings("ignore")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    df_train = pd.read_csv("data/final_train.csv")
    df_test = pd.read_csv("data/final_test.csv")
    logging.info("Data loaded")

    X_train = df_train.drop(columns=["booking_status"])
    y_train = df_train["booking_status"]
    X_test = df_test.drop(columns=["booking_status"])
    y_test = df_test["booking_status"]

    model = LogisticRegression(random_state=1)
    logging.info("Model instance created")

    model.fit(X_train, y_train)
    logging.info("Model fitted")

    y_pred = model.predict(X_test)
    logging.info("Predictions made")

    logging.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    logging.info("Classification report:\n%s", classification_report(y_test, y_pred))

    with open("src/models/logistic_regression_baseline.pkl", "wb") as f:
        pickle.dump(model, f)
    logging.info("Model saved")


if __name__ == "__main__":
    main()
