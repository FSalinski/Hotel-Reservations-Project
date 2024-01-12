import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from bin.feature_engineering import (
    CyclicScaler,
    CategoricalEncoder,
    Scaler,
    ColumnAdder,
    TargetTransformer,
)


def main():
    """
    Feature engineering script
    Pipeline:
    1. One hot encoding "type_of_meal_plan" and "market_segment_type" columns
    2. Ordinal encoding "room_type_reserved" column
    3. Cyclical encoding date columns
    4. Adding new columns
    5. Scaling numerical columns into [0, 1] range
    6. Turning target column into binary and reordering columns
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    df_train = pd.read_csv("data/cleaned_train.csv")
    df_test = pd.read_csv("data/cleaned_test.csv")
    logging.info("Data loaded")

    logging.info("Sample of train data:\n%s", df_train.head())
    logging.info("Parameters of train data:\n%s", df_train.describe())
    logging.info("Columns:\n%s", df_train.columns)

    categorical_encoder = CategoricalEncoder(
        cols_to_one_hot_encode=["type_of_meal_plan", "market_segment_type"],
        cols_to_ordinal_encode=["room_type_reserved"],
    )

    cyclic_scaler = CyclicScaler(
        cols_to_scale=["arrival_date", "arrival_month", "arrival_year"]
    )

    column_adder = ColumnAdder()

    scaler = Scaler(
        cols_to_scale=[
            "lead_time",
            "no_of_adults",
            "no_of_children",
            "no_of_weekend_nights",
            "no_of_week_nights",
            "no_of_previous_cancellations",
            "no_of_previous_bookings_not_canceled",
            "no_of_special_requests",
            "avg_price_per_room",
            "total_guests",
            "total_price",
        ],
        scaler="minmax",
    )

    target_transformer = TargetTransformer(target_col="booking_status")

    pipeline = Pipeline(
        [
            ("categorical_encoder", categorical_encoder),
            ("cyclic_scaler", cyclic_scaler),
            ("column_adder", column_adder),
            ("scaler", scaler),
            ("target_transformer", target_transformer),
        ]
    )

    pipeline.fit(df_train)
    df_train = pipeline.transform(df_train)
    df_test = pipeline.transform(df_test)

    logging.info("Sample of train data after feature engineering:\n%s", df_train.head())
    logging.info(
        "Parameters of train data after feature engineering:\n%s", df_train.describe()
    )
    logging.info("Sample of test data after feature engineering:\n%s", df_test.head())
    logging.info(
        "Parameters of test data after feature engineering:\n%s", df_test.describe()
    )
    logging.info("Columns after feature engineering:\n%s", df_train.columns)
    logging.info("Data info after feature engineering:\n%s", df_train.info())

    logging.info("Saving data...")
    df_train.to_csv("data/final_train.csv", index=False)
    df_test.to_csv("data/final_test.csv", index=False)


if __name__ == "__main__":
    main()
