import logging
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    """
    Script for splitting data into training and testing sets
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_csv("data/data_cleaned.csv")
    logging.info("Data loaded")

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    df_train.to_csv("data/cleaned_train.csv", index=False)
    df_test.to_csv("data/cleaned_test.csv", index=False)
    logging.info("Data saved")


if __name__ == "__main__":
    main()
