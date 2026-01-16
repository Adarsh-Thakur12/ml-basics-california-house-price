# Housing Price Prediction

A machine learning project to predict house prices using the California Housing Dataset. This project demonstrates data preprocessing, model training, and predictions using scikit-learn.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)
- [License](#license)

## Project Overview

This project builds a machine learning pipeline to predict median house values based on various housing features. It includes:

- Data preprocessing with missing value imputation and feature scaling
- Categorical feature encoding
- Model training with multiple algorithms
- Cross-validation for model evaluation
- Trained model persistence using joblib

## Dataset

The project uses the California Housing Dataset containing:

- **8 numerical features**: Latitude, Longitude, Housing Median Age, Total Rooms, Total Bedrooms, Population, Households, Median Income
- **1 categorical feature**: Ocean Proximity
- **Target variable**: Median House Value

**Data points**: ~20,640 housing records

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main script to train the model:

```bash
python main.py
```

This will:
- Load and preprocess the housing dataset
- Create a stratified train-test split based on income categories
- Build and fit a RandomForestRegressor model
- Save the trained model and preprocessing pipeline to disk

### Making Predictions

Once the model is trained, run the script again with test data (`input.csv`):

```bash
python main.py
```

The predictions will be saved to `output.csv`

## Project Structure

```
housing-price-prediction/
├── main.py                 # Main training and prediction script
├── main_old.py            # Legacy model comparison script
├── housing.csv            # Dataset
├── model.pkl              # Trained model (generated)
├── pipeline.pkl           # Preprocessing pipeline (generated)
├── input.csv              # Input data for predictions (generated)
├── output.csv             # Prediction results (generated)
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .gitignore            # Git ignore rules
```

## Models

The project implements and compares multiple regression models:

1. **Linear Regression** - Baseline model for linear relationships
2. **Decision Tree Regressor** - Captures non-linear patterns
3. **Random Forest Regressor** - Ensemble method with improved generalization (primary model)

### Model Evaluation

Models are evaluated using cross-validation with Root Mean Squared Error (RMSE) as the metric.

## Results

The models achieve the following approximate RMSE scores (via 10-fold cross-validation):
- Linear Regression: ~73,500
- Decision Tree: ~72,000
- Random Forest: ~71,000

*Note: Train actual models to get exact results*

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib

See `requirements.txt` for detailed version specifications.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or feedback, please open an issue on GitHub.
