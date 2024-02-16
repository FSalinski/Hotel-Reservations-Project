"""
Script for training Random Forest model using best parameters found in random_forest_gs.py
We'll train the model on the entire training set
"""

import logging
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    df_train = pd.read_csv("data/final_train.csv")

    X_train = df_train.drop(columns=["booking_status"])
    y_train = df_train["booking_status"]

    model = RandomForestClassifier(criterion="gini", max_depth=16, max_features="sqrt", n_estimators=150, random_state=1)   
    logging.info("Model instance created")

    model.fit(X_train, y_train)
    logging.info("Model fitted")

    with open("src/models/random_forest_final.pkl", "wb") as f:
        pickle.dump(model, f)
    logging.info("Model saved")
    

if __name__ == "__main__":
    main()
