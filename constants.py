# Dataset Constants
DATASET_NAME = "data/actual_dataset.csv"
DATASET_SPLIT_DATE = "2017-5-7"

# Forecasting Model Hyperparameters
SEQ_LENGTH = 24
FEATURE_DIM = 77
FORECAST_HORIZON = 1
TRAIN_NUM_EPOCHS = 30
BATCH_SIZE = 32

# Gradient Estimation Hyperparameters
MODEL_NAME = 'output/load_forecasting_model_weights.h5'
ALPHA = 0.01
GRAD_NUM_EPOCHS = 10
BETA = 0.9
DELTA = 0.05
BOUND = 0.4
TEMPERATURE_VARIATION = 5
