# %%
import tensorflow as tf
import numpy as np
# %%
class NaiveDense:
	# Acts as dense layers similar to tf.keras.layers.Dense
	def __init__(self, input_size, output_size, activation):
		self.activation = activation

		# Create a 2D tensor using input and output size
		w_shape = (input_size, output_size)
		# Then initialize it with random values between 0 and 0.1
		w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
		# A tf.Variable is a specific kind of tensor meant to hold a mutable state
		self.W = tf.Variable(w_initial_value)

		# Init the scaling vector to be zeros
		b_shape = (output_size,)
		b_initial_value = tf.zeros(b_shape)
		self.b = tf.Variable(b_initial_value)

	def __call__(self, inputs):
		# Forward pass
		return self.activation(tf.matmul(inputs, self.W) + self.b)

	@property
	def weights(self):
		# This will allow us to retrieve layer's weights
		return [self.W, self.b]
# %%
class NaiveSequential:
	# Create a sequential model to chain layers similar to tf.keras.models.Sequential
	def __init__(self, layers):
		self.layers = layers
	def __call__(self, inputs):
		x = inputs
		for layer in self.layers:
			x = layer(x)
		return x
	
	@property
	def weights(self):
		# Allow one to easily keep track of the layers parameters
		weights = []
		for layer in self.layers:
			weights += layer.weights
		return weights
# %%
# Using this NaiveDense class and this NaiveSequential class, we can create a mock Keras model
model = NaiveSequential([
	NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
	NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4
# %%
class BatchGenerator:
	def __init__(self, images, labels, batch_size=128):
		self.index = 0
		self.images = images
		self.labels = labels
		self.batch_size = batch_size
	def next(self):
		images = self.images[self.index : self.index + self.batch_size]
		labels = self.labels[self.index : self.index + self.batch_size]
		self.index += self.batch_size
		return images, labels
# %%
def one_training_step(model, images_batch, labels_batch):
	# In this training step, we need to
		# 1. Compute the predictions of the model for the images in the batch 
		# 2. Compute the loss value for these predictions given the actual labels
		# 3. Compute the gradient of the loss with regard to the model’s weights
		# 4. Move the weights by a small amount in the direction opposite to the gradient
	with tf.GradientTape() as tape:
		predictions = model(images_batch)
		per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
			labels_batch, predictions)
		average_loss = tf.reduce_mean(per_sample_losses)
	gradients = tape.gradient(average_loss, model.weights)
	update_weights(gradients, model.weights)
	return average_loss
# %%
learning_rate = 1e-3
def update_weights(gradients, weights):
	# The simplest way to implement this update_weights function is to subtract gradient * learning_rate from each weight:
	for g, w in zip(gradients, model.weights):
		# assign_sub is the equivalent of -= for TensorFlow variables.
		w.assign_sub(w * learning_rate)
# %%
def fit(model, images, labels, epochs, batch_size=128):
	for epoch_counter in range(epochs):
		print('Epoch %d' % epoch_counter)
		batch_generator = BatchGenerator(images, labels)
		for batch_counter in range(len(images) // batch_size):
			images_batch, labels_batch = batch_generator.next()
			loss = one_training_step(model, images_batch, labels_batch)
			if batch_counter % 100 == 0:
				print('loss at batch %d: %.2f' % (batch_counter, loss))
# %%
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
fit(model, train_images, train_labels, epochs=10, batch_size=128)
# %%
# We can evaluate the model by taking the argmax of its predictions over the test images, and comparing it to the expected labels:
predictions = model(test_images)
# Calling .numpy() on a TensorFlow tensor converts it to a NumPy tensor.
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print('accuracy: %.2f' % matches.mean())
# %%
