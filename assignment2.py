from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the training dataset
df = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

# Define target variable and features
target = df["meal"]
features = df.drop(["meal", "id", "DateTime"], axis=1)

# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.33, random_state=42)

# Initialize and fit the DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)
modelFit = model.fit(train_X, train_Y)

# Calculate in-sample and out-of-sample accuracy
train_accuracy = accuracy_score(train_Y, model.predict(train_X))
print("\n\nIn-sample accuracy: %s%%\n\n" % str(round(100 * train_accuracy, 2)))

test_accuracy = accuracy_score(test_Y, model.predict(test_X))
print("\n\nOut-of-sample accuracy: %s%%\n\n" % str(round(100 * test_accuracy, 2)))

# Load the test dataset for predictions
test_df = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
new_test = test_df.drop(["meal", "id", "DateTime"], axis=1)

# Make predictions on the test dataset
pred = model.predict(new_test)

# Store predictions in the test dataset
test_df["predicted_meal"] = pred
print(test_df[["id", "predicted_meal"]].head())

# Save the predictions to a CSV file
test_df[["id", "predicted_meal"]].to_csv("meal_predictions.csv", index=False)

print("\nPredictions have been saved to 'meal_predictions.csv'.")
