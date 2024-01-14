import pandas as pd
import numpy as np
import pytz
import datetime as dt
import constants as const


def load_dataset(dataset_name, dataset_split_date, verbose=1):
    # read dataset
    df = pd.read_csv(dataset_name, sep=";")
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)
    df = df.drop(['entsoe'], axis=1)

    print(f'Dataset shape: {df.shape}')

    # train test split based on dateset_split_date
    loc_tz = pytz.timezone('Europe/Zurich')
    split_date = loc_tz.localize(
        dt.datetime(int(dataset_split_date.split("-")[0]), int(dataset_split_date.split("-")[1]),
                    int(dataset_split_date.split("-")[2]), 0, 0, 0, 0))
    train_df = df.loc[(df.index <= split_date)].copy()
    test_df = df.loc[df.index > split_date].copy()

    print(f'Training dataset shape: {train_df.shape}')
    if verbose==1:
        print("Training dataset:")
        print(train_df.head(5))
    print(f'Test dataset shape: {test_df.shape}')
    if verbose==1:
        print("Testing dataset:")
        print(test_df.head(5))

    return train_df, test_df


def prepare_labelled_dataset(df):
    actual_load = np.array(df['actual_load'].copy(), dtype=float)
    data = np.array(df, dtype=float)

    X = []
    y = []

    # prepare labelled dataset for sequential data
    for i in range(len(data) - const.SEQ_LENGTH - const.FORECAST_HORIZON):
        X.append(data[i:i + const.SEQ_LENGTH])
        y_new = actual_load[i + const.SEQ_LENGTH: i + const.SEQ_LENGTH + const.FORECAST_HORIZON].reshape(-1,const.FORECAST_HORIZON)
        y.append(y_new)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1, const.FORECAST_HORIZON)

    return X, y
