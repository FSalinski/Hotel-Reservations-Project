import logging
import pandas as pd
from bin.cleaning import ColumnDropper, DateCleaner
from sklearn.pipeline import Pipeline


def main():
    '''
    Data cleaning script
    Pipeline:
    1. Drop columns
    2. Change incorrect 29.02 date to 01.03
    '''
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_csv("data/data.csv")
    logging.info("Data loaded")
    logging.info("Sample of data:\n%s", df.head())
    logging.info("Parameters of data:\n%s", df.describe())

    coldropper = ColumnDropper(cols_to_drop=["Booking_ID"])
    datecleaner = DateCleaner()

    pipeline = Pipeline([("coldropper", coldropper), ("datecleaner", datecleaner)])

    df = pipeline.fit_transform(df)
    logging.info("Data cleaned")
    logging.info("Sample of data:\n%s", df.head())
    logging.info("Parameters of data:\n%s", df.describe())

    df.to_csv("data/data_cleaned.csv", index=False)
    logging.info("Data saved")


if __name__ == "__main__":
    main()
