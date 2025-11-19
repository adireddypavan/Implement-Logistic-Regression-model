# Implement-Logistic-Regression-model
python program to Implement  Logistic Regression model for given dataset

# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# Step 2: Create a sample dataset
# Example: Predict whether a person buys a product (1 = Yes, 0 = No)
data = pd.DataFrame({
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 45, 34],
    'Salary': [20000, 25000, 50000, 52000, 48000, 60000, 58000, 62000, 52000, 42000],
    'Purchased': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
})

print("----- Dataset -----")
print(data, "\n")

# Step 3: Define independent (X) and dependent (y) variables
X = data[['Age', 'Salary']]
y = data['Purchased']

# Step 4: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Create and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate model performance
print("----- Model Evaluation -----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Display Actual vs Predicted
result = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("----- Actual vs Predicted -----")
print(result, "\n")

# Step 9: Predict for a new input
new_data = pd.DataFrame({'Age': [30, 50], 'Salary': [30000, 55000]})
new_pred = model.predict(new_data)
print("----- New Predictions -----")
print(pd.concat([new_data, pd.Series(new_pred, name='Purchased_Prediction')], axis=1))
