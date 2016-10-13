# -*- coding: utf-8 -*-y
import numpy as np
import cPickle as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import sys

##############################################################

x_train_file = sys.argv[1]
y_train_file = sys.argv[2]
x_test_file = sys.argv[3]
y_test_file = sys.argv[4]
predictions_file = sys.argv[5]
trained_model_file = sys.argv[6]

# load training data
with open(y_train_file, "rb") as f:
    y_train = pickle.load(f)
X_train = np.load(x_train_file)

# fit scaler on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
parameters = {'C': [0.001,  0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], 'penalty': ['l2']}

# get trained model with C that leads to best f1-score
optimal_C = GridSearchCV(LogisticRegression(), parameters, cv=5, refit=True)

optimal_C.fit(X_train, y_train)

print "Grid scores on development set:"
for score in optimal_C.grid_scores_:
    print score
print "Optimal C on development set:"
print optimal_C.best_params_
print

# load test data and transform it
with open(y_test_file, "rb") as f:
    y_test = pickle.load(f)
X_test = np.load(x_test_file)
X_test = scaler.transform(X_test)
y_pred = optimal_C.predict(X_test)

print "Scores on test set:"
# print several evaluation metrics
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print "Accuracy: " + str(accuracy)
print "Precision: " + str(precision)
print "Recall: " + str(recall)
print "F1: "+ str(f1)

# write predictions on test set to file
with open(predictions_file, "wb") as f:
    pickle.dump(y_pred, f)
print "Predictions saved"

# write trained estimator to file
joblib.dump(optimal_C, trained_model_file)
print "Trained model saved"
