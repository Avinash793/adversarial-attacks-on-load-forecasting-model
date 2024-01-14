# Adversarial Attacks on Load Forecasting Model

## Dataset Information
### Dataset Used
1. ENTSO-E Dataset (European Network of Transmission System Operators for Electricity) for hourly load data in Switzerland.
2. DarkSky Dataset for hourly temperature and weather Icon information of 8 major cities in Switzerland.

### Dataset Raw Features
There are 21 raw features at each timestamp:
1. **Load**
2. **8 Cities temperature**
3. **8 Cities weather Icon Information** - categorical feature tells which weather icon [categories: icon1, icon2, icon3]
4. **Holiday** - boolean feature tells weather holiday in switzerland on that date
5. **Month** - categorical feature tells data of which month  [categories: Jan, Feb, ... , Dec]
6. **Day** - categorical feature tells data of which day [categories: Mon, Tues, Wed, Thrus, Fri, Sat, Sun]
7. **Hour** - categorical feature tells data of which hour  [categories: 0, 1, 2, ... , 23]

### Dataset Source
You can use already preprocessed data present in `data` folder with name `actual_dataset.csv` .

**Feature Vector 77 dimensional at each timestamp:** \
actual_load - 1 feature \
8 cities temperature - 8 features \
8 cities weather icon one hot encoding - (8 cities x 3 categories of icon) = 24 features \
holiday - 1 feature \
weekday one hot encoding  - 7 features \
hour one hot encoding  - 24 features \
month one hot encoding  - 12 features 

**NOTE:** Please ignore `entsoe` feature column in `actual_dataset.csv`. 


## Train Load Forecasting Model
1. change `DATASET_SPLIT_DATE` in `constants.py` according to how you want to split train and test dataset.
2. Simply Run
    ```shell
    python forecasting.py
    ```
3. It will save trained model weights in `output/load_forecasting_model_weights.h5`. save `output/loss_epoch_curve.png` and `output/actual_predicted_load.png` images.


## Generate Adversarial Datasets
1. Simply Run:
    ```shell
    python adversarial.py
    ```
2. It will generate adversarial datasets for various temperature variation in `data` folder. For Ex: `adversarial_dataset_temp_1.csv` means generate adversarial temperature dataset with 1 Fahrenheit change in temperature.


## Results
Check `results.ipynb` file to see various plots like:
1. Temperature Profile
2. Load Forecasting Profile
3. Forecasting MAPE with Temperature Variation