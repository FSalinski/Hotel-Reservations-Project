import logging
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    """
    Script for splitting training data into training and validation sets
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_csv("data/final_train.csv")
    logging.info("Data loaded")

    df_train, df_val = train_test_split(df, test_size=0.3, random_state=1)

    df_train.to_csv("data/final_train_train.csv", index=False)
    df_val.to_csv("data/final_train_val.csv", index=False)
    logging.info("Data saved")


if __name__ == "__main__":
    main()
