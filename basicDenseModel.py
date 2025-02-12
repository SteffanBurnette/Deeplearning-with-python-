#Basic imports
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing the basic mnist datset via tflow datasets
from tensorflow.keras.datasets import mnist
from tensorflow import keras 
from tensorflow.keras import layers

#This is destructured into 4 numpy arrays
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"The test set: {test_images.shape}")
print(f"The test set labels: {len(test_labels)}")

#Creating the fully connected neural network
model = keras.Sequential([
 layers.Dense(512, activation="relu"),
 layers.Dense(10, activation="softmax")
])

#Compiling the model
model.compile(optimizer="rmsprop",
 loss="sparse_categorical_crossentropy",
 metrics=["accuracy"] )

#Preprocessing the data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255


#Training the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

#Getting predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]