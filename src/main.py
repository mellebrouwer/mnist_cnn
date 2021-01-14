import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers

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

# Reshape 3D to 2D
X = X.reshape((len(X), 28*28), order='F')

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# ---------------------------------
# SIMPLE SVM BASELINE
# ---------------------------------

model = SGDClassifier(random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
mcm = multilabel_confusion_matrix(y_test, preds)

print("\nSVM BASELINE")
print(f"Accuracy: {acc}")

# ---------------------------------
# SIMPLE NEURAL NET BASELINE
# ---------------------------------

# Input shape & random state
tf.random.set_seed(0)
_, input_shape = X_train.shape

# Build model
model = keras.Sequential([
                          keras.Input(shape=(input_shape,)),
                          layers.Dense(128, activation='relu'),
                          layers.Dense(128, activation='relu'),
                          layers.Dense(10, activation="softmax"),
                          ])

# Compile model
model.compile(
             optimizer="rmsprop",
             loss="sparse_categorical_crossentropy",
             metrics=["sparse_categorical_accuracy"],
             )

# Set early stopping
early_stop = [keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-2,
                    patience=3,
                    verbose=1,
                    )]
    
# Train model
history = model.fit(
                    X_train,
                    y_train,
                    batch_size=64,
                    epochs=30,
                    callbacks=early_stop,
                    validation_split=0.2,
                    )

# Test model
preds = model.predict(X_test)

# Convert softmax probabilities to integers
preds = np.argmax(preds,axis=-1)

acc = accuracy_score(y_test, preds)
mcm = multilabel_confusion_matrix(y_test, preds)

print("\nNEURAL NET BASELINE")
print(f"Accuracy: {acc}")

# ---------------------------------------------
# PREPARE & SPLIT DATASETS RESHAPE TO ndim=4
# ---------------------------------------------

# Reassign original arrays
X = imagearray.reshape(-1, 28, 28, 1)
y = labelarray

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# ---------------------------------
# CONVOLUTIONAL NEURAL NETWORK
# ---------------------------------

# Input shape & random state
tf.random.set_seed(0)

# Build model
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile model
model.compile(
             optimizer="rmsprop",
             loss="sparse_categorical_crossentropy",
             metrics=["sparse_categorical_accuracy"],
             )

# Set early stopping
early_stop = [keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-2,
                    patience=3,
                    verbose=1,
                    )]
    
# Train model
history = model.fit(
                    X_train,
                    y_train,
                    batch_size=64,
                    epochs=30,
                    callbacks=early_stop,
                    validation_split=0.2,
                    )

# Test model
preds = model.predict(X_test)

# Conver softmax probabilities to integers
preds = np.argmax(preds,axis=-1)

acc = accuracy_score(y_test, preds)
mcm = multilabel_confusion_matrix(y_test, preds)

print("\nCONVOLUTIONAL NEURAL NETWORK")
print(f"Accuracy: {acc}")

