import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import constants as const
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dataset import load_dataset, prepare_labelled_dataset
import warnings

warnings.filterwarnings("ignore")


# LSTM Based RNN load forecasting model
def forecasting_model(seq_length, feature_dim, output_dim):
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_length, feature_dim)))
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Dense(output_dim))
    return model


# train model using Adam optimizer and MAE loss function
def train_model(model, X_train, y_train):
    model.compile(loss="mean_absolute_error", optimizer="adam")
    history = model.fit(X_train, y_train, epochs=const.TRAIN_NUM_EPOCHS, batch_size=const.BATCH_SIZE)
    # save model weights
    model.save_weights('output/load_forecasting_model_weights.h5')

    return model, history


# model training diagnostics
def training_diagnostics(model, history, X_train, y_train, X_test, y_test):
    # plot training loss with epochs curve
    loss = history.history["loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('output/loss_epoch_curve.png')

    # evaluate model mae on training dataset
    train_mae = model.evaluate(X_train, y_train, verbose=0)
    print(f'mean absolute error on training dataset: {train_mae}')

    # evaluate model mae on test dataset
    test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'mean absolute error on test dataset: {test_mae}')


def predict(model, X_test, y_test, load_scaler):
    # predict on test dataset
    y_pred = model.predict(X_test, verbose=0)
    print(f'y_pred dataset shape: {y_pred.shape}')
    print(f'y_test dataset shape: {y_test.shape}')

    # rescale predicted and actual load using train dataset standard scaler
    y_pred_df = pd.DataFrame(y_pred, columns=[['load']])
    y_actual_df = pd.DataFrame(y_test, columns=[['load']])
    y_pred = load_scaler.inverse_transform(y_pred_df)
    y_actual = load_scaler.inverse_transform(y_actual_df)

    # plot actual vs predicted load
    plt.figure()
    plt.plot(y_actual, label='actual', color='b')
    plt.plot(y_pred, label='predicted', color='r')
    plt.title('Actual vs Predicted Load')
    plt.xlabel("Hour")
    plt.ylabel("Load(MW)")
    plt.legend()
    plt.savefig('output/actual_predicted_load.png')


def train_forecasting_model():
    # load train and test dataset
    train_df, test_df = load_dataset(const.DATASET_NAME, const.DATASET_SPLIT_DATE)

    # preprocess train and test dataset
    non_categorical_features = ['bsl_t', 'brn_t', 'zrh_t', 'lug_t', 'lau_t', 'gen_t', 'stg_t', 'luz_t']
    features_scaler = StandardScaler()
    train_df[non_categorical_features] = features_scaler.fit_transform(train_df[non_categorical_features])
    test_df[non_categorical_features] = features_scaler.transform(test_df[non_categorical_features])
    load_scaler = StandardScaler()
    train_df['actual_load'] = load_scaler.fit_transform(train_df[['actual_load']])
    test_df['actual_load'] = load_scaler.transform(test_df[['actual_load']])
    print("Training dataset after preprocessing:")
    print(train_df.head(5))
    print("Test dataset after preprocessing:")
    print(test_df.head(5))

    X_train, y_train = prepare_labelled_dataset(train_df)
    print(f'X_train dataset shape: {X_train.shape}')
    print(f'y_train dataset shape: {y_train.shape}')
    # print(X_train[0:2])
    # print(y_train[0:2])

    X_test, y_test = prepare_labelled_dataset(test_df)
    print(f'X_test dataset shape: {X_test.shape}')
    print(f'y_test dataset shape: {y_test.shape}')
    # print(X_test[0:2])
    # print(y_test[0:2])

    model = forecasting_model(const.SEQ_LENGTH, const.FEATURE_DIM, const.FORECAST_HORIZON)
    model, history = train_model(model, X_train, y_train)
    training_diagnostics(model, history, X_train, y_train, X_test, y_test)
    predict(model, X_test, y_test, load_scaler)


if __name__ == "__main__":
    train_forecasting_model()
