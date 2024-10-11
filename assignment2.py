from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

target = df["meal"]
features = df.drop(["meal", "id", "DateTime"], axis=1)

train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.33, random_state=42)

tree_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)
fitted_tree_model = tree_model.fit(train_X, train_Y)

train_accuracy = accuracy_score(train_Y, fitted_tree_model.predict(train_X))
print("\n\nIn-sample accuracy: %s%%\n\n" % str(round(100 * train_accuracy, 2)))

test_accuracy = accuracy_score(test_Y, fitted_tree_model.predict(test_X))
print("\n\nOut-of-sample accuracy: %s%%\n\n" % str(round(100 * test_accuracy, 2)))

test_df = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
new_test = test_df.drop(["meal", "id", "DateTime"], axis=1)

test_predictions = fitted_tree_model.predict(new_test)

test_df["predicted_meal"] = test_predictions
print(test_df[["id", "predicted_meal"]].head())

test_df[["id", "predicted_meal"]].to_csv("meal_predictions.csv", index=False)
