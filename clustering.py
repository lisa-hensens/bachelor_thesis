import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random


class Clustering:
    def __init__(self, train_data, train_y, test_data, test_y, all_labels):
        """
        :param train_data: training data as Dataframe
        :param train_y: training labels as Series
        :param test_data: test data as Dataframe
        :param test_y: test labels as Series
        :param all_labels: ndarray that consists of all possible labels available in the data
        """
        self.train_data = train_data
        self.train_y = train_y.to_numpy()
        self.test_data = test_data
        self.test_y = test_y.to_numpy()
        self.labels = all_labels

    def transform_labels(self, predict):
        """
        Transform the numerical labels into their corresponding labels as mentioned in the data

        :param predict: list with numerical labels
        :return: list with corresponding categorical labels

        """
        new_predicted = []
        for j in predict:
            new_predicted.append(self.labels[j])
        return new_predicted

    def transform_names(self, y):
        """
        Transforms all categorical labels as they are mentioned in the data into their numerical labels

        :param y: list with categorical labels
        :return: list with corresponding numerical labels
        """
        new_names = []
        for i in y:
            for j in self.labels:
                if i == j:
                    new_names.append(np.where(self.labels == j)[0][0])
        return new_names

    def get_scorings(self, y_true, y_pred):
        """
        Computes scoring(s) of the given predicted and true labels.
        In this case: accuracy and F1-score.

        :param y_true: list of lists with actual labels
        :param y_pred: list of lists with predicted
        :return: list of scorings accompanied by matching names of scoring
        """
        accuracy = []
        f1 = []
        for i in range(len(y_true)):
            accuracy.append(metrics.accuracy_score(y_true, y_pred))
            f1.append(metrics.f1_score(y_true, y_pred, average="weighted"))
        return [accuracy, f1], ["Accuracy", "F1-score"]

    def run_kmeans(self):
        """
        Runs K-Means, prints the performance scores and plots the corresponding confusion matrix.

        :return: none
        """
        kmean = KMeans(n_clusters=len(self.labels), random_state=random.seed())
        predicted = kmean.fit_predict(self.test_data)
        y_pred = self.transform_labels(predicted)
        y_true = self.test_y
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.labels)
        values, method = self.get_scorings(y_true, y_pred)
        for i in range(len(values)):
            print(method[i], np.mean(values[i]))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)
        disp.plot()
        plt.xticks(rotation = 40)
        plt.show()

    def run_agglo(self):
        """
        Runs agglomerative clustering algorithm, prints the performance scores
        and plots the corresponding confusion matrix.

        :return: none
        """
        aggl = AgglomerativeClustering(n_clusters=len(self.labels))
        predicted = aggl.fit_predict(self.test_data)
        y_pred = self.transform_labels(predicted)
        y_true = self.test_y
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.labels)
        values, method = self.get_scorings(y_true, y_pred)
        for i in range(len(values)):
            print(method[i], np.mean(values[i]))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)
        disp.plot()
        plt.xticks(rotation=40)
        plt.show()

