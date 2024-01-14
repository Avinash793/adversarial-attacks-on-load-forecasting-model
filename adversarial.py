import numpy as np
import pandas as pd
import constants as const
from dataset import load_dataset
from forecasting import forecasting_model
from sklearn.preprocessing import StandardScaler


# bound generated adversarial temperature features - attacker capabilities
def check_constraint(x_orig, x_new):
    for temp_idx in range(1, 9):
        x_new[temp_idx] = np.clip(x_new[temp_idx], x_orig[temp_idx] - const.BOUND * np.abs(x_orig[temp_idx]),
                                  x_orig[temp_idx] + const.BOUND * np.abs(x_orig[temp_idx]))
    return x_new


# calculate signed gradient
def calculate_signed_gradient(X, temp_idx, model):
    X_plus = X.copy()
    X_minus = X.copy()
    X_plus[0, const.SEQ_LENGTH-1, temp_idx] += const.DELTA
    X_minus[0, const.SEQ_LENGTH-1, temp_idx] -= const.DELTA
    gradient = model.predict(X_plus, verbose=0) - model.predict(X_minus, verbose=0)
    return np.sign(gradient)


# Black Box Based Gradient Estimation Algorithm to generate hard to detect adversarial dataset
def gradient_estimation(df, model, temp_variation):
    X_adversarial = []
    data = np.array(df, dtype=float)
    alpha = const.ALPHA * temp_variation

    # loop over all datapoint
    for i in range(len(data)):
        epoch = 1
        if i < const.SEQ_LENGTH - 1:
            X_adversarial.append(data[i])
            continue

        # gamma which denotes to increase or decrease load, if 0 then increase load, if 1 then decrease load
        gamma = np.random.randint(2)
        X = data[i - const.SEQ_LENGTH + 1:i + 1].reshape(1, const.SEQ_LENGTH, const.FEATURE_DIM)

        # optimize GRAD_NUM_EPOCHS times
        while epoch <= const.GRAD_NUM_EPOCHS:
            for temp_idx in range(1, 9):
                signed_gradient = calculate_signed_gradient(X, temp_idx, model)
                if gamma == 0:
                    X[0][const.SEQ_LENGTH - 1][temp_idx] += alpha * signed_gradient
                else:
                    X[0][const.SEQ_LENGTH - 1][temp_idx] -= alpha * signed_gradient
            epoch = epoch + 1

        # bound temperature data - attacker capabilities
        X[0][const.SEQ_LENGTH - 1] = check_constraint(data[i], X[0][const.SEQ_LENGTH - 1])
        # store generated adversarial datapoint
        X_adversarial.append(X[0][const.SEQ_LENGTH - 1])

    return pd.DataFrame(X_adversarial, index=df.index, columns=df.columns)


def generate_adversarial_datasets():
    # load train and test dataset
    train_df, test_df = load_dataset(const.DATASET_NAME, const.DATASET_SPLIT_DATE)

    # load saved forecasting model
    model = forecasting_model(const.SEQ_LENGTH, const.FEATURE_DIM, const.FORECAST_HORIZON)
    model.load_weights(const.MODEL_NAME)
    print("loaded trained forecasting model ...")

    # preprocess test dataset based on train dataset standard scaler
    non_categorical_features = ['actual_load', 'bsl_t', 'brn_t', 'zrh_t', 'lug_t', 'lau_t', 'gen_t', 'stg_t', 'luz_t']
    features_scaler = StandardScaler()
    features_scaler.fit(train_df[non_categorical_features])
    test_df[non_categorical_features] = features_scaler.transform(test_df[non_categorical_features])

    # generate adversarial datasets for various variation in temperature (in Fahrenheit)
    for i in range(1, const.TEMPERATURE_VARIATION + 1):
        adversarial_df = gradient_estimation(test_df, model, temp_variation=i)
        # rescale features to actual values using train data standard scaler
        adversarial_df[non_categorical_features] = features_scaler.inverse_transform(adversarial_df[non_categorical_features])
        # save generated adversarial dataset
        adversarial_df.to_csv('data/adversarial_dataset_temp_' + str(i) + ".csv")
        print(f'Generated adversarial dataset for temperature variation={i}F')


if __name__ == "__main__":
    generate_adversarial_datasets()