from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd

df = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
target = df["meal"]
features = df.drop(["meal", "id", "DateTime"], axis=1)

# Handling class imbalance by setting class_weight
class_weights = class_weight.compute_class_weight('balanced', classes=[0, 1], y=target)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.33, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, class_weight=class_weight_dict, random_state=42)
modelFit = model.fit(train_X, train_Y)

print(f"\n\nIn-sample accuracy: {round(100 * accuracy_score(train_Y, model.predict(train_X)), 2)}%\n\n")
print(f"\n\nOut-of-sample accuracy: {round(100 * accuracy_score(test_Y, model.predict(test_X)), 2)}%\n\n")

test_df = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
new_test = test_df.drop(["meal", "id", "DateTime"], axis=1)
pred = model.predict(new_test)

test_df["predicted_meal"] = pred
print(test_df[["id", "predicted_meal"]].head())

test_df[["id", "predicted_meal"]].to_csv("meal_predictions.csv", index=False)
