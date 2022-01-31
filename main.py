from clustering import Clustering
from dataanalysis import DataAnalysis
from rulebased import RuleBased
from preprocessing_data import PreprocessingData
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC

# Parameters
drop_variables = False
C_value = 3

# Data preparation
filename = "./data.pkl"
preprocess = PreprocessingData(filename)
pickle_data = preprocess.get_pickle_data()
X, y = preprocess.get_transformed_data(pickle_data)
all_labels = np.unique(y)

# Filter out specific labels from the data set
if drop_variables:
    eliminate_indeces = y.index[y == 'spop_1'].tolist()
    eliminate_indeces.extend(y.index[y == 'spop_2'].tolist())
    eliminate_indeces.extend(y.index[y == 'kdm3b'].tolist())
    eliminate_indeces.extend(y.index[y == 'yy1'].tolist())
    eliminate_indeces.extend(y.index[y == 'cltc'].tolist())
    eliminate_indeces.extend(y.index[y == 'ppm1d'].tolist())
    eliminate_indeces.extend(y.index[y == 'dyrk1a'].tolist())
    eliminate_indeces.extend(y.index[y == 'pacs1'].tolist())
    X.drop(eliminate_indeces, inplace=True)
    y.drop(eliminate_indeces, inplace=True)

# Dealing with class imbalance for clustering
train_symptoms, train_labels = preprocess.get_balanced_data(X, y)
df = X.merge(train_symptoms, how='left', indicator=True)
indices = df.index[df['_merge'] == 'left_only']
test_symptoms = X.iloc[indices]
test_labels = y.iloc[indices]

# Stratified data split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, stratify=y, random_state=0)
df = X.merge(X_train, how='left', indicator=True)
indices = df.index[df['_merge'] == 'left_only']
pickle_data.drop(indices.values, axis=0, inplace=True)

# Programs
clust = Clustering(X_train, y_train, X_test, y_test, all_labels)
da = DataAnalysis(pickle_data)
expl = RuleBased(all_labels)

# Run program 1
clust.run_kmeans()
clust.run_agglo()

# Run program 2
da.remove_duplicates()
da.filter_severity()
da.filter_basic_symptoms()
for i in all_labels:
    print("Genetic disorder: ", i)
    a, b, frequents = da.get_frequent_symptoms(i)
    keys, counts, combinations = da.get_counts(i, frequents)
    print("Keys: ", keys)
    print("Counts: ", counts)
    print(combinations)

# Run program 2b
y_true = pickle_data['label']
y_pred = []
pred = []
indices = []
counter = 0
for index, patient in enumerate(pickle_data['hpo_all_name']):
    label = expl.get_label(patient)
    pred.append(label)
    if label != 'Undiagnosed':
        y_pred.append(label)
    else:
        indices.append(index)
        counter += 1
print("Amount of patients that have been undiagnosed: ", counter)

pred = [expl.get_label(patient) for patient in pickle_data['hpo_all_name']]
true_labels = pickle_data['label']
print("Accuracy of program 2b without dropping: ", metrics.accuracy_score(true_labels, pred))
print("F1 of program 2b without dropping: ", metrics.f1_score(true_labels, pred, average="weighted"))

y_true = y_true.drop(y_true.index[indices]).tolist()
print("Accuracy of program 2b: ", metrics.accuracy_score(y_true, y_pred))
print("F1 of program 2b: ", metrics.f1_score(y_true, y_pred, average="weighted"))

clf = DecisionTreeClassifier()
model = clf.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
acc = metrics.accuracy_score(y_pred_test, y_test)
f1 = metrics.f1_score(y_pred_test, y_test, average="weighted")
print("Accuracy of decision tree classifier: ", acc)
print("F1 of decision tree classifier: ", f1)
plt.figure(figsize=(10, 15))
tree.plot_tree(clf, feature_names=X_train.columns.values, class_names=all_labels)
plt.show()
text_representation = tree.export_text(clf, feature_names=X_train.columns.values.tolist())
print(text_representation)

classifier = SVC(C=C_value)
mod = classifier.fit(X_train, y_train)
y_hat_test = mod.predict(X_test)
acc2 = metrics.accuracy_score(y_hat_test, y_test)
f1_2 = metrics.f1_score(y_hat_test, y_test, average="weighted")
print("Accuracy of Support Vector Classifier : {}".format(acc2))
print("F1 of Support Vector Classifier : {}".format(f1_2))
