import numpy as np
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from math import sqrt


def wrapper_for_decision_tree_in_sklearn(X, y, current_state_to_predict):
	clf = DecisionTreeClassifier()
	clf.fit(X, y)
	current_state_to_predict = np.array(current_state_to_predict).reshape(1,-1)
	predicted_state = clf.predict(current_state_to_predict)
	return predicted_state


def wrapper_for_decision_tree_accuracy(X, y, relative_test_size):
	X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=relative_test_size, random_state=42)


	clf = DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	score = accuracy_score(pred,y_test)

	return score


class RileyRandomForest(object):
    def __init__(self, n_estimators = 10):
        self.n_estimators = n_estimators
        self.trees = {}
        self.predictions = []
        self.prediction = []

    def forest_fit(self,X,y):
        for i in range(self.n_estimators):
            self.trees["tree{}".format(i)] = DecisionTreeClassifier(max_features='auto')
            self.trees["tree{}".format(i)].fit(X, y)

    def forest_predict(self,X):
        lst = []
        for tree in self.trees:
            self.predictions.append(self.trees[tree].predict(X))
        for i,_ in enumerate(self.predictions[0]):
            for pred in self.predictions:
                lst.append(pred[i])
            final = max(set(lst), key=lst.count)
            lst = []
            self.prediction.append(final)
        return self.prediction

class RileySVMForest(object):
    def __init__(self, n_estimators = 10):
        self.n_estimators = n_estimators
        self.trees = {}
        self.predictions = []
        self.prediction = []

    def forest_fit(self,X,y):
        for i in range(self.n_estimators):
            self.trees["tree{}".format(i)] = DecisionTreeClassifier(max_features='auto')
            self.trees["tree{}".format(i)].fit(X, y)
            if i % 5 == 0:
                self.trees["SVM{}".format(i)] = SVC()
                self.trees["SVM{}".format(i)].fit(X, y)

    def forest_predict(self,X):
        lst = []
        for tree in self.trees:
            self.predictions.append(self.trees[tree].predict(X))
        for i,_ in enumerate(self.predictions[0]):
            for pred in self.predictions:
                lst.append(pred[i])
            if i < 5:
                print lst
            final = max(set(lst), key=lst.count)
            lst = []
            self.prediction.append(final)
        return self.prediction


def wrapper_for_riley_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42)
    clf = RileyRandomForest()
    clf.forest_fit(X_train, y_train)
    pred = clf.forest_predict(X_test)
    score = accuracy_score(pred, y_test)

    return score









