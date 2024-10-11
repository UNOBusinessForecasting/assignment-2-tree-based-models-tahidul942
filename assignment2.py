# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the training dataset
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

# Display the first few rows of the dataset
print(data.head())

# Initialize the DecisionTreeClassifier with parameters
model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)

# Define target variable (Y) and features (X), dropping unnecessary columns
Y = data["meal"]
X = data.drop(["meal", "id", "DateTime"], axis=1)

# Split the data into training and testing sets (33% test size)
x, xt, y, yt = train_test_split(X, Y, test_size=0.33, random_state=42)

# Fit the decision tree model on the training data
modelFit = model.fit(x, y)

# Evaluate in-sample (training data) accuracy
in_sample_accuracy = accuracy_score(y, model.predict(x))
print("\n\nIn-sample accuracy: %s%%\n\n" % str(round(100 * in_sample_accuracy, 2)))

# Evaluate out-of-sample (test data) accuracy
out_of_sample_accuracy = accuracy_score(yt, model.predict(xt))
print("\n\nOut-of-sample accuracy: %s%%\n\n" % str(round(100 * out_of_sample_accuracy, 2)))

# Load the test dataset for predictions
test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

# Drop unnecessary columns from the test dataset
testNew = test.drop(["meal", "id", "DateTime"], axis=1)

# Make predictions on the test dataset
test_predictions = model.predict(testNew)

# Store predictions (binary 0 or 1) in the test dataset
test["meal_predictions"] = test_predictions

# Output the first few rows of the test dataset with predictions
print(test[["id", "meal_predictions"]].head())

# Save the predictions to a CSV file if needed
test[["id", "meal_predictions"]].to_csv("meal_predictions.csv", index=False)

print("\nPredictions have been saved to 'meal_predictions.csv'.")
