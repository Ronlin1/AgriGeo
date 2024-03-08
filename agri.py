# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

# Load dataset from the CSV file
data = pd.read_csv('predicting_food_crises_data.csv')  

# Define the features (X) and the target variable (y)
# Replace these with the actual column names from your dataset
features = ['year', 'month', 'ndvi_mean', 'rain_mean', 'et_mean', 'acled_count', 'p_staple_food', 'area', 'cropland_pct', 'pop', 'ruggedness_mean', 'pasture_pct']
target_variable = 'fews_ipc'  

X = data[features]

# Handle missing values in the target variable
imputer = SimpleImputer(strategy='mean')
y = imputer.fit_transform(data[target_variable].values.reshape(-1, 1))
X_imputed = imputer.fit_transform(X)

# Split the data into a training set and a testing set (e.g., 80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Log metrics to MLflow
mlflow.log_param("features", features)
mlflow.log_param("target_variable", target_variable)
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2", r2)

# Log model to MLflow
mlflow.sklearn.log_model(model, "model")

# Save the model using joblib
joblib.dump(model, 'agrigeo_model.pkl')

# Save the model path for future reference
model_path = "agrigeo_model.pkl"

# Log the model path to MLflow
mlflow.log_artifact(model_path)
