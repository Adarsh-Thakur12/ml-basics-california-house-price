import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# Load the Data Set
housing = pd.read_csv("housing.csv")

# Create the Stratified Shuffle based on income category
housing['income_cat'] = pd.cut(
    housing['median_income'],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Work on the Copy of the train data
housing = strat_train_set.copy()

# Seperate Features and Labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# List Numerical and Categorical Data
num_attributs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attributs = ["ocean_proximity"]

# Make the pipeline
# For Numerical Data
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler()),

])
# For Categorical Data
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
# Full Pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributs),
    ("cat", cat_pipeline, cat_attributs)
])

# Trandform the Data
housing_prepared = full_pipeline.fit_transform(housing)

# Train the Model
# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_pred)
# print(f"The Root Mean Square Error For Linear Regression is {lin_rmse} ")
lin_rmses = -cross_val_score(lin_reg, housing_prepared,
                             housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_rmses).describe())

# Decision Tree
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_pred = dec_reg.predict(housing_prepared)
# dec_rmse =root_mean_squared_error(housing_labels,dec_pred)
# print(f"The Root Mean Square Error For Decision Tree is {dec_rmse}")
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10
                             )
print(pd.Series(dec_rmses).describe())

# Random Forest Tree
random_reg = RandomForestRegressor()
random_reg.fit(housing_prepared, housing_labels)
random_pred = random_reg.predict(housing_prepared)
# random_rmse = root_mean_squared_error(housing_labels, random_pred)
# print(f"The Root Mean Square Error For Random Forest Tree is {random_rmse}")
random_rmses = -cross_val_score(random_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10
                             )
print(pd.Series(random_rmses).describe())
