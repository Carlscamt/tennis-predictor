
# Tennis Match Prediction and Betting System Documentation

This document provides a comprehensive overview of the project, its structure, and how to use it. It aims to clarify the purpose of each file and provide a clear workflow for data processing, model training, and prediction.

## 1. Project Overview

This project is a machine learning system designed to predict the outcomes of professional tennis matches. It includes functionalities for data collection, feature engineering, model training, and making predictions through an interactive interface. It also features several betting strategies to simulate betting based on the model's predictions.

## 2. File Structure and Descriptions

The project is organized into several key areas. Here is a breakdown of the most important files and their roles:

### Core Application

- **`match_predictor.py`**: The main entry point for the interactive application. It uses Streamlit to create a user interface where you can select players, a surface, and get match predictions and betting advice.
- **`tennis_model_fixed.joblib`**: The trained and saved XGBoost model that is used for predictions.
- **`surface_encoder.joblib`**: A scikit-learn encoder that is saved and used to transform the "surface" feature into a numerical format for the model.

### Data Pipeline

These files are responsible for getting and preparing the data for the model.

- **`tennis_data/`**: This directory is the designated location for all raw and processed data.
- **`tennis_data/download_data.py`**: A script to automatically download historical tennis data from `tennis-data.co.uk` for the years 2000-2025.
- **`tennis_data/process_data.py`**: This script takes the downloaded Excel files, combines them into a single file, and saves it as `tennis_data.csv`.
- **`feature_engineering_fixed.py`**: A crucial script that takes the raw `tennis_data.csv` and engineers a rich set of features for the model. It correctly handles temporal dependencies to avoid data leakage and saves the final, feature-rich dataset to `tennis_data_features_fixed.csv`.

### Modeling

These files are used to train and evaluate the prediction model.

- **`train_model_fixed.py`**: The primary script for training the model. It uses time-series cross-validation for robust evaluation and saves the final trained model to `tennis_model_fixed.joblib`.
- **`model_trainer.py`**: A utility class for more detailed model evaluation, including betting simulations during cross-validation. This is useful for analyzing the model's performance from a betting perspective.

### Betting Strategies

These modules define the logic for the different betting strategies.

- **`betting_strategy_v2.py`**: An adaptive betting strategy based on the Kelly Criterion. It adjusts bet sizes based on the model's confidence and estimated odds.
- **`betting_strategy_uncertainty.py`**: An alternative betting strategy that incorporates the model's uncertainty into its calculations, making it more conservative when the model is less certain.

### Backtesting

These scripts are for testing the performance of the betting strategies on historical data.

- **`backtest_2025.py`**: A script to backtest the `AdaptiveBettingStrategy` on data from the year 2025.
- **`backtest_2025_uncertainty.py`**: A script to backtest the `UncertaintyShrinkageBetting` strategy on 2025 data.
- **`backtesting_engine_optimized.py`**: A more generalized backtesting engine that can be used to test strategies over different time periods and with different parameters.

### Drafts and Unused Files

These files appear to be drafts, old versions, or otherwise not in use. They can likely be **safely removed** to clean up the project:

- **`betting_strategy.py`**: Empty file.
- **`feature_engineering_optimized.py`**: Empty file.
- **`train_model.py`**: Empty file.
- **`test_features.py`**: Empty file.
- **`http___tennis-data.co.uk_alldata.php.htm`** and **`http___tennis-data.co.uk_alldata.php_files/`**: A saved webpage, not needed for the project to run.
- **`Tennis Betting Machine Learning Strategies_.pdf`**: A reference document, not part of the application code.

## 3. Setup and Usage Workflow

Here is the recommended workflow to get the project up and running:

### Step 1: Install Dependencies

Make sure you have all the required Python libraries installed. You can do this by running:

```bash
pip install -r requirements.txt
```

### Step 2: Download the Data

Run the `download_data.py` script to fetch all the historical data. This will download a series of Excel files into the `tennis_data/` directory.

```bash
python tennis_data/download_data.py
```

### Step 3: Process the Raw Data

Next, run the `process_data.py` script to combine all the downloaded Excel files into a single CSV file named `tennis_data.csv`.

```bash
python tennis_data/process_data.py
```

### Step 4: Engineer Features

Now, run the `feature_engineering_fixed.py` script. This will take the `tennis_data.csv` file, calculate all the features, and create the final dataset `tennis_data_features_fixed.csv` that the model needs.

```bash
python feature_engineering_fixed.py
```

### Step 5: Train the Model

With the features ready, you can train the model by running the `train_model_fixed.py` script. This will create the `tennis_model_fixed.joblib` and `surface_encoder.joblib` files.

```bash
python train_model_fixed.py
```

### Step 6: Run the Prediction App

Now you are ready to start the interactive Streamlit application:

```bash
streamlit run match_predictor.py
```

This will open a page in your web browser where you can select players and a surface to get predictions.

## 4. Backtesting the Strategies

If you want to evaluate the performance of the betting strategies, you can run the backtesting scripts. For example, to backtest the uncertainty-based strategy for the year 2025, you would run:

```bash
python backtest_2025_uncertainty.py
```

This will output performance metrics like ROI and final bankroll, and it will save a plot of the performance to `backtest_analysis_2025.png`.
