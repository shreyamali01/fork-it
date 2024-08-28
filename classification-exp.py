import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import accuracy, precision, recall, mae, rmse
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

#for training
X_train = pd.DataFrame(X[0:70,:])
y_train = pd.Series(y[0:70], dtype = "category")
print(X_train)

#for testing
X_test = pd.DataFrame(X[70:,:])
y_test = pd.Series(y[70:], dtype = "category")
print(X_test)

criterion = ['information_gain','gini_index']

for criteria in criterion:
    model = DecisionTree(criterion=criteria)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    model.plot()

    accuracy_val = accuracy(y_test, y_pred)
    precision_val = precision(y_test, y_pred, cls=0)
    recall_val = recall(y_test, y_pred, cls=0)

    print(f'Accuracy: {accuracy_val}')
    print(f'Precision: {precision_val}')
    print(f'Recall: {recall_val}')


