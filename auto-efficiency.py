import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error


np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "model year", "origin", "car name"])

# Data Cleaning
# Replace '?' with NaN in 'horsepower' and convert it to numeric
data.replace("?", np.nan, inplace=True)
data['horsepower'] = pd.to_numeric(data['horsepower'])

# Drop rows with missing values in 'horsepower'
data.dropna(subset=['horsepower'], inplace=True)

# Drop the 'car name' column
data.drop(columns=["car name"], inplace=True)

# Convert categorical features to numeric
data['origin'] = data['origin'].astype('category').cat.codes
data['model year'] = data['model year'].astype('category').cat.codes
data['cylinders'] = data['cylinders'].astype('category').cat.codes

# Define features and target
X = data.drop("mpg", axis=1)
y = (data["mpg"] > 23).astype(int)  # Binary classification: above or below 23 mpg

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Implementing and evaluating the custom decision tree
custom_tree = DecisionTreeClassifier()
custom_tree.fit(X_train, y_train)
y_pred_custom = custom_tree.predict(X_test)

# Implementing and evaluating scikit-learn's decision tree
sklearn_tree = DecisionTreeClassifier()
sklearn_tree.fit(X_train, y_train)
y_pred_sklearn = sklearn_tree.predict(X_test)

# Calculate metrics for custom decision tree
accuracy_custom = accuracy_score(y_test, y_pred_custom)
precision_custom = precision_score(y_test, y_pred_custom)
recall_custom = recall_score(y_test, y_pred_custom)
mse_custom = mean_squared_error(y_test, y_pred_custom)

# Calculate metrics for scikit-learn's decision tree
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
precision_sklearn = precision_score(y_test, y_pred_sklearn)
recall_sklearn = recall_score(y_test, y_pred_sklearn)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

# Print results
print(f"Custom Decision Tree Accuracy: {accuracy_custom:.2f}")
print(f"Custom Decision Tree Precision: {precision_custom:.2f}")
print(f"Custom Decision Tree Recall: {recall_custom:.2f}")
print(f"Custom Decision Tree MSE: {mse_custom:.2f}")

print(f"scikit-learn Decision Tree Accuracy: {accuracy_sklearn:.2f}")
print(f"scikit-learn Decision Tree Precision: {precision_sklearn:.2f}")
print(f"scikit-learn Decision Tree Recall: {recall_sklearn:.2f}")
print(f"scikit-learn Decision Tree MSE: {mse_sklearn:.2f}")



