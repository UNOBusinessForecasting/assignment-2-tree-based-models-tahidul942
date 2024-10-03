
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


TrainData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
TrainData.head()


from sklearn.model_selection import train_test_split

Y = TrainData['meal']
X = TrainData.drop(columns=['meal','id','DateTime'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state= 42, class_weight = "balanced")
modelFit = model.fit(x_train,y_train)

# Test our model using the testing data
predict = model.predict(x_test)
acc1 = accuracy_score(y_test, predict)

print("Model accuracy is {}%.".format(acc1*100))

# %%
TestData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
xt = TestData.drop(columns = ['meal','id','DateTime'], axis =1)
pred = model.predict(xt)
pred

