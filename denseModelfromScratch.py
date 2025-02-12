'''A simple Python class, NaiveDense, that creates two TensorFlow
variables, W and b, and exposes a __call__() method that applies the preceding
transformation.
'''
import tensorflow as tf
class NaiveDense:
 def __init__(self, input_size, output_size, activation): #On instantiation the layer takes in the input size. output size and activation function
  self.activation = activation
  w_shape = (input_size, output_size)  #Creates a matrix, W, of shape (input_size, output_size), initialized with random values.
  w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1) #Creates random initial weight values
  self.W = tf.Variable(w_initial_value) #Sets the random intial weight values

  b_shape = (output_size,) #Create a vector, b, of shape (output_size,), initialized with zeros. 
  b_initial_value = tf.zeros(b_shape) #Initializes the bias vector with zeros
  self.b = tf.Variable(b_initial_value) #Capsulates the values into a tensor

 def __call__(self, inputs): #Applys the forward pass
  return self.activation(tf.matmul(inputs, self.W) + self.b) # ActivationFunction(X^T * W + b)

 @property
 def weights(self):           #Convenience method for retrieving the layer’s weights
  return [self.W, self.b]

class NaiveSequential:
  def __init__(self, layers):#Takes the layers as input
    self.layers = layers 

  def __call__(self, inputs): #Takes the inputs and assign it to x
    x = inputs
    for layer in self.layers: #Loops through the layers and have the inputs and outputs passed to the next layers (forward propagation)
      x = layer(x)
    return x

  @property
  def weights(self):
    weights = []
    for layer in self.layers:
      weights += layer.weights #Returns all the learned weights
    return weights

model = NaiveSequential([
 NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
 NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
]) 
assert len(model.weights) == 4

import math
# Math.ceil() static method always rounds up and returns the smallest integer greater than or equal to a given number.

class BatchGenerator:
 def __init__(self, images, labels, batch_size=128):
  assert len(images) == len(labels) #Makes sure that the # of images = the # of labels
  self.index = 0 #Will be used to split up the batches
  self.images = images
  self.labels = labels
  self.batch_size = batch_size
  self.num_batches = math.ceil(len(images) / batch_size) #Calculates the number of batches needed

def next(self): #Loads in the current batch and has the index go to the next batch afterwards
  images = self.images[self.index : self.index + self.batch_size]
  labels = self.labels[self.index : self.index + self.batch_size]
  self.index += self.batch_size
  return images, labels

def one_training_step(model, images_batch, labels_batch):
  with tf.GradientTape() as tape: #Run the “forward pass” (compute the model’s predictions under a GradientTape scope)
    predictions = model(images_batch) 
    per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy( 
    labels_batch, predictions) 
    average_loss = tf.reduce_mean(per_sample_losses) 
    gradients = tape.gradient(average_loss, model.weights)  #Compute the gradient of the loss with regard to the weights. The output gradientsis a list where each entry corresponds to a weight from the model.weights list.
    update_weights(gradients, model.weights) #Update the weights using the gradients
  return average_loss

from tensorflow.keras import optimizers
optimizer = optimizers.SGD(learning_rate=1e-3)
def update_weights(gradients, weights):
  optimizer.apply_gradients(zip(gradients, weights))

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255 
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
fit(model, train_images, train_labels, epochs=10, batch_size=128)

import numpy as np
predictions = model(test_images)
predictions = predictions.numpy() 
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")