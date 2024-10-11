from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

# Define features and target variable
X = data.drop(columns=["meal", "id", "DateTime"])
y = data["meal"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize the Decision Tree model
model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=15, random_state=42)

# Fit the model on the training data
modelFit = model.fit(X_train, y_train)

# Calculate in-sample accuracy (training set)
in_sample_accuracy = accuracy_score(y_train, model.predict(X_train))
print(f"In-sample accuracy: {in_sample_accuracy * 100:.2f}%")

# Calculate out-of-sample accuracy (testing set)
out_of_sample_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Out-of-sample accuracy: {out_of_sample_accuracy * 100:.2f}%")

# Load the test dataset for predictions
test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
X_new = test.drop(columns=["meal", "id", "DateTime"])

# Show the first few rows of the new test data
X_new.head()

# Make predictions on the new data
predictions = modelFit.predict(X_new)

# Optional: Store predictions in a DataFrame or CSV file
predictions_df = pd.DataFrame(predictions, columns=["Predicted Meal"])
predictions_df.to_csv("meal_predictions.csv", index=False)
