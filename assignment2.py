from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)

Y = data["meal"]
X = data.drop(["meal", "id", "DateTime"], axis=1)

x, xt, y, yt = train_test_split(X, Y, test_size=0.33, random_state=42)

modelFit = model.fit(x, y)

print("\n\nIn-sample accuracy: %s%%\n\n" % str(round(100 * accuracy_score(y, model.predict(x)), 2)))
print("\n\nOut-of-sample accuracy: %s%%\n\n" % str(round(100 * accuracy_score(yt, model.predict(xt)), 2)))

test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
testNew = test.drop(["meal", "id", "DateTime"], axis=1)

pred = modelFit.predict(testNew)
pred = [int(p) for p in pred]

print(pred)
