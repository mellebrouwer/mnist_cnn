import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

# ---------------------------------
# IMPORT MNIST DATASET
# ---------------------------------

imagefile = '/Users/mellebrouwer/Programmeren/Github/mnist_cnn/data/t10k-images-idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

labelfile = '/Users/mellebrouwer/Programmeren/Github/mnist_cnn/data/t10k-labels-idx1-ubyte'
labelarray = idx2numpy.convert_from_file(labelfile)

X = imagearray
y = labelarray

# ---------------------------------
# PREPARE & SPLIT DATASETS
# ---------------------------------

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Reshape 3D to 2D
X_train = X_train.reshape((len(X_train), 28*28), order='F')
X_test = X_test.reshape((len(X_test), 28*28), order='F')

# ---------------------------------
# SIMPLE SVM BASELINE
# ---------------------------------

model = SGDClassifier(random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
mcm = multilabel_confusion_matrix(y_test, preds)

print(mcm.mean(axis=0))
print(acc)