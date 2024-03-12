# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib ejee


# Handle missing values in the target variable
imputer = SimpleImputer(strategy='mean')
y = imputer.fit_transform(data[target_variable].values.reshape(-1, 1))
X_imputed = imputer.fit_transform(X)