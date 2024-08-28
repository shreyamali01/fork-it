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

#Using nested cross-validation to find the optimal depth

criterion = ['information_gain','gini_index']
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X = pd.DataFrame(X)
y = pd.Series(y)
K = 5
m = 4
ind_Xy = int(len(X)/K)
hyper_paramaters = []
for i in range(K):
    X_test = pd.DataFrame(X[i*ind_Xy : (i+1)*ind_Xy]).reset_index(drop=True)
    y_test = pd.Series(y[i*ind_Xy : (i+1)*ind_Xy]).reset_index(drop=True)
    X_outerTrain = pd.concat( (X[0:i*ind_Xy], X[(i+1)*ind_Xy:]), axis=0 ).reset_index(drop=True)
    y_outerTrain = pd.concat( (y[0:i*ind_Xy], y[(i+1)*ind_Xy:]), axis=0 ).reset_index(drop=True)
    fin_accuracy = None
    fin_depth = None
    best_criteria = None
    for criteria in criterion:
        for depth in depths:
            accuracy_sum = 0
            for j in range(m):
                X_validation = pd.DataFrame(X_outerTrain[j*ind_Xy:(j+1)*ind_Xy]).reset_index(drop=True)
                y_validation = pd.Series(y_outerTrain[j*ind_Xy:(j+1)*ind_Xy]).reset_index(drop=True)
                X_innerTrain = pd.concat( (X_outerTrain[0:j*ind_Xy], X_outerTrain[(j+1)*ind_Xy:]), axis=0 ).reset_index(drop=True)
                y_innerTrain = pd.concat( (y_outerTrain[0:j*ind_Xy], y_outerTrain[(j+1)*ind_Xy:]), axis=0 ).reset_index(drop=True)
                model = DecisionTree(criterion = criteria, max_depth = depth)
                model.fit(X_innerTrain, y_innerTrain)
                y_pred = model.predict(X_validation)
                accuracy_sum += accuracy(y_pred, y_validation)
            avg_accuracy = accuracy_sum / m
            if ((fin_accuracy is None) or (avg_accuracy > fin_accuracy)):
                fin_accuracy = avg_accuracy
                fin_depth = depth
                best_criteria = criteria
    fin_model = DecisionTree(criterion = best_criteria, max_depth = fin_depth)
    fin_model.fit(X_outerTrain, y_outerTrain)
    y_pred = fin_model.predict(X_test)
    hyper_paramaters.append((fin_depth, best_criteria, accuracy(y_pred, y_test)))


total_accuracy = 0
for i in range(K):
    total_accuracy += hyper_paramaters[i][2]
avg_accuracy_kfolds = float(total_accuracy)/5
print(hyper_paramaters)
print(avg_accuracy_kfolds)