from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
data = pd.read_csv(data_url)

X = data.drop(columns=["meal", "id", "DateTime"])
y = data["meal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)
modelFit = model.fit(X_train, y_train)

in_sample_accuracy = accuracy_score(y_train, model.predict(X_train))
out_of_sample_accuracy = accuracy_score(y_test, model.predict(X_test))

test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

X_new = test_data.drop(columns=["meal", "id", "DateTime"])

pred = modelFit.predict(X_new)

predictions_df = pd.DataFrame(pred, columns=['meal'])
predictions_df.to_csv("meal_predictions.csv", index=False)
