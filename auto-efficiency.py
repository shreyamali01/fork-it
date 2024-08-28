import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import rmse, mae
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Reading the data
url = '/Users/shreya/Documents/fork-it/auto-mpg.data'
data = pd.read_csv(url, sep=r'\s+', header=None,
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "model year", "origin", "car name"])


# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

data = data.drop('car name',axis=1) #irrelevant data to us

data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['horsepower']=data['horsepower'].fillna(data['horsepower'].mean(), inplace=True)

data = pd.get_dummies(data, columns=['origin'], drop_first=True)

X = data.drop('mpg', axis=1)
y = data['mpg']

#preprocessing

data_proc = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data_proc.iloc[:, 1:]
y = data_proc.iloc[:, 0]

X = X.astype(np.float64)
y = y.astype(np.float64)

#splitting the data
a = len(y) 
index = int(0.7*a)

X_train= X[:index]
X_test = X[index:]
y_train = y[:index]
y_test = y[index:]

#performance on our decision tree
model_my = DecisionTree(max_depth=5)
model_my.fit(X_train,y_train)
y_pred = model_my.predict(X_test)

rmse_val = rmse(y_pred,y_test)
mae_val = mae(y_pred,y_test)

print('On our decision tree RMSE: {}'.format(rmse_val))
print('On our decision tree MAE: {}'.format(mae_val))

#performance on decision tree module from sckit

model1 = DecisionTreeRegressor(max_depth=5,random_state =0)
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)

#calculating performance metrics
rmse_model1 = rmse(y_pred1,y_test)
mae_model1 = mae(y_pred1, y_test)

print('Scikit-learn Decision Tree RMSE: {}'.format(rmse_model1))
print('Scikit-learn Decision Tree MAE: {}'.format(mae_model1))