"""
Housing Price Prediction Model

This module trains a machine learning model to predict house prices using
the California Housing Dataset. It includes data preprocessing, model training,
and prediction capabilities.

Author: Adarsh Singh Sengar
Date: 2026
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Constants
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
DATA_FILE = "housing.csv"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"


def build_pipeline(num_attributes, cat_attributes):
    """
    Build a preprocessing pipeline for numerical and categorical features.
    
    Args:
        num_attributes (list): List of numerical feature column names
        cat_attributes (list): List of categorical feature column names
        
    Returns:
        ColumnTransformer: Configured preprocessing pipeline
    """
def build_pipeline(num_attributes, cat_attributes):
    """
    Build a preprocessing pipeline for numerical and categorical features.
    
    Args:
        num_attributes (list): List of numerical feature column names
        cat_attributes (list): List of categorical feature column names
        
    Returns:
        ColumnTransformer: Configured preprocessing pipeline
    """
    # For Numerical Data: impute missing values and standardize
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler()),
    ])
    
    # For Categorical Data: one-hot encode
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Combine both pipelines
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)
    ])
    return full_pipeline


def train_model():
    """
    Train and save the machine learning model.
    
    This function:
    1. Loads the housing dataset
    2. Splits data using stratified sampling based on income categories
    3. Preprocesses features
    4. Trains a RandomForestRegressor model
    5. Saves the model and pipeline for later use
    """
    print("Loading dataset...")
    housing = pd.read_csv(DATA_FILE)

    # Create income categories for stratified splitting
    housing['income_cat'] = pd.cut(
        housing['median_income'],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # Stratified train-test split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        # Save test set to file for predictions
        housing.loc[test_index].drop("income_cat", axis=1).to_csv(INPUT_FILE, index=False)
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    # Separate features and labels
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    # Identify numerical and categorical attributes
    num_attributes = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]

    # Build and fit preprocessing pipeline
    print("Building preprocessing pipeline...")
    preprocessing_pipeline = build_pipeline(num_attributes, cat_attributes)
    housing_prepared = preprocessing_pipeline.fit_transform(housing_features)

    # Train RandomForest model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)
    model.fit(housing_prepared, housing_labels)

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocessing_pipeline, PIPELINE_FILE)
    print("Model and pipeline saved successfully!")
    print(f"Congratulations! Model is Trained and saved to {MODEL_FILE}")


def make_predictions():
    """
    Load trained model and make predictions on input data.
    
    This function:
    1. Loads the pre-trained model and preprocessing pipeline
    2. Transforms input data using the saved pipeline
    3. Makes predictions
    4. Saves results to output file
    """
    print("Loading trained model and pipeline...")
    model = joblib.load(MODEL_FILE)
    preprocessing_pipeline = joblib.load(PIPELINE_FILE)

    print("Loading input data...")
    input_data = pd.read_csv(INPUT_FILE)
    
    print("Making predictions...")
    transformed_data = preprocessing_pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    
    # Add predictions to input data
    input_data["predicted_median_house_value"] = predictions

    # Save results
    input_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        train_model()
    else:
        make_predictions()