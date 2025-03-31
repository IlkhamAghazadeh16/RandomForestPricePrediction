import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
file_path = "taxi_trip_pricing.csv"  # Update this path if needed
data = pd.read_csv(file_path)

# Define target and features
target = 'Trip_Price'  # Update this if your target column has a different name
features = data.drop(columns=[target])  # Drop the target column to get features
target_values = data[target]  # Extract the target column

# Identify numerical and categorical columns
num_cols = features.select_dtypes(include=['float64', 'int64']).columns
cat_cols = features.select_dtypes(include=['object']).columns

# Define preprocessing for numerical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', StandardScaler())                 # Standardize numerical data
])

# Define preprocessing for categorical data
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Convert categories to one-hot encoding
])

# Combine numerical and categorical pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# Apply preprocessing to the dataset
X = preprocessor.fit_transform(features)

# Fill missing target values with the mean
y = target_values.fillna(target_values.mean())

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Preprocessing complete. Ready for training!")

# Training of the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R²: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# saving the model

import joblib

# Save the trained model to a file
model_file = "taxi_price_model.pkl"
joblib.dump(model, model_file)
print(f"Model saved to {model_file}")

# prediction function

def predict_taxi_price(input_data, preprocessor, model):
    """
    Predict taxi fare based on input features.

    Args:
    input_data (dict): Input parameters as a dictionary.
    preprocessor: Preprocessing pipeline.
    model: Trained model.

    Returns:
    float: Predicted taxi fare.
    """
    import pandas as pd

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Align columns with training features
    required_columns = preprocessor.feature_names_in_
    for col in required_columns:
        if col not in input_df:
            input_df[col] = None  # Add missing columns with placeholder values

    # Preprocess the input data
    processed_data = preprocessor.transform(input_df)

    # Predict the fare
    predicted_price = model.predict(processed_data)
    return predicted_price[0]

# Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train, y_train)
print(f"Best parameters found: {grid_search.best_params_}")

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred_best = best_model.predict(X_test)

# Calculate metrics for the best model
best_mae = mean_absolute_error(y_test, y_pred_best)
best_mse = mean_squared_error(y_test, y_pred_best)
best_rmse = np.sqrt(best_mse)
best_mape = mean_absolute_percentage_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)

# Display evaluation metrics for the best model
print("\nEvaluation Metrics for Best Model:")
print(f"Best Model R²: {best_r2:.2f}")
print(f"Best Model Mean Absolute Error (MAE): {best_mae:.2f}")
print(f"Best Model Mean Squared Error (MSE): {best_mse:.2f}")
print(f"Best Model Root Mean Squared Error (RMSE): {best_rmse:.2f}")
print(f"Best Model Mean Absolute Percentage Error (MAPE): {best_mape:.2f}%")


# Load model and preprocessor
model = joblib.load("taxi_price_model.pkl")
preprocessor = joblib.load("taxi_price_preprocessor.pkl")

test_cases = [
    {"Trip_Distance_km": 5.0, "Day_of_Week": "Weekday", "Time_of_Day": "Morning", "Weather": "Clear","Base_Fare": 2.5, "Passenger_Count": 2, "Per_Km_Rate": 1.5, "Trip_Duration_Minutes": 10, "Per_Minute_Rate": 0.2, "Traffic_Conditions": "Low"},
    {"Trip_Distance_km": 15.0, "Day_of_Week": "Weekday", "Time_of_Day": "Evening", "Weather": "Snow", "Base_Fare": 2.0,"Passenger_Count": 3, "Per_Km_Rate": 2.0, "Trip_Duration_Minutes": 25, "Per_Minute_Rate": 0.3, "Traffic_Conditions": "Low"},
    {"Trip_Distance_km": 8.0, "Day_of_Week": "Weekday", "Time_of_Day": "Afternoon", "Weather": "Clear", "Base_Fare": 2.7,"Passenger_Count": 1, "Per_Km_Rate": 1.7, "Trip_Duration_Minutes": 15, "Per_Minute_Rate": 0.25, "Traffic_Conditions": "Medium"},
    {"Trip_Distance_km": 3.0, "Day_of_Week": "Weekend", "Time_of_Day": "Morning", "Weather": "Rain", "Base_Fare": 3.4 ,"Passenger_Count": 4, "Per_Km_Rate": 1.8, "Trip_Duration_Minutes": 8, "Per_Minute_Rate": 0.22, "Traffic_Conditions": "High"}
]

for case in test_cases:
    predicted_price = predict_taxi_price(case, preprocessor, model)
    print(f"Test Case: {case}")
    print(f"Predicted Taxi Price: {predicted_price:.2f}\n")

import matplotlib.pyplot as pltno

# Creating a scatter plot to compare actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')

# Adding a red dashed line for perfect predictions (where actual = predicted)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

# Adding labels and title
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Taxi Prices')
plt.legend()

# Show the plot
plt.show()
