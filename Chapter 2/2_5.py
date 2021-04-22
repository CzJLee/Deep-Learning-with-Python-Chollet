# Chapter 2.5
# Building a simple network "from scratch" using tensorflow
# Build a model for the MNIST dataset

# %%
import tensorflow as tf
import numpy as np

# %%
# Let's build a simple model first using keras, then using the naive approach

# First, import the dataset.
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the dataset into 2D tensors
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# %%
# Construct the model using Keras Sequential model, and Dense layers
model_keras = tf.keras.models.Sequential([
	tf.keras.layers.Dense(512, activation="relu"),
	tf.keras.layers.Dense(10, activation="softmax")
])

# %%
# Then complie the model and fit the data
model_keras.compile(optimizer = "rmsprop", 
	loss = "sparse_categorical_crossentropy",
	metrics = ["accuracy"])
	# Remember that metrics needs to be an array

model_keras.fit(train_images, train_labels, epochs = 5, batch_size = 128)



# %%
# Now lets build our naive model
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
		# This method makes a NaiveDense object callable
		return self.activation(tf.matmul(inputs, self.W) + self.b)

	@property
	def weights(self):
		# This will allow us to retrieve layer's weights
		# E.g. through layer.weights
		return [self.W, self.b]
# %%
class NaiveSequential:
	# Create a sequential model to chain layers similar to tf.keras.models.Sequential
	def __init__(self, layers):
		# Pass in an array of layers.
		self.layers = layers
	def __call__(self, inputs):
		# Make each layer callable. 
		x = inputs
		for layer in self.layers:
			# For each layer in the sequential model, apply the callable function of NaiveDense using the previous input, then return the value
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
# Initialize a model using our Naive models
model = NaiveSequential([
	NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
	NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
# Here we create two dense layers. Notice that we have to specify the input and output size. 
# We then pass in the specific activation functions we want to use. 
# Note that in Keras, something like "relu" gets expanded to tf.nn.relu
assert len(model.weights) == 4
# %%
class BatchGenerator:
	# Create an iterator that will iterate over all of the images and labels, in sizes of batch_size.
	def __init__(self, images, labels, batch_size=128):
		self.index = 0
		self.images = images
		self.labels = labels
		self.batch_size = batch_size
	def next(self):
		# I believe that this functions the same as __next__
		images = self.images[self.index : self.index + self.batch_size]
		labels = self.labels[self.index : self.index + self.batch_size]
		self.index += self.batch_size
		return images, labels
# %%
# In this training step, we need to
		# 1. Compute the predictions of the model for the images in the batch 
		# 2. Compute the loss value for these predictions given the actual labels
		# 3. Compute the gradient of the loss with regard to the model’s weights
		# 4. Move the weights by a small amount in the direction opposite to the gradient

def one_training_step(model, images_batch, labels_batch):
	with tf.GradientTape() as tape:
		# https://www.tensorflow.org/api_docs/python/tf/GradientTape
		# Make predictions based on the current model
		predictions = model(images_batch)

		# Compute the loss based on the true labels and predictions made
		per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
			labels_batch, predictions)

		# Compute the average_loss
		# https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
		average_loss = tf.reduce_mean(per_sample_losses)

	# Calculate the gradient, then call our update_weights function that adjusts the weights using the gradient and learning rate.
		gradients = tape.gradient(average_loss, model.weights)
		update_weights(gradients, model.weights)
	return average_loss
# %%
# Define a set learning rate.
learning_rate = 1e-3

def update_weights(gradients, weights):
	# The simplest way to implement this update_weights function is to subtract gradient * learning_rate from each weight:
	# In reality, we would use an optimizer from tf.keras.optimizers
	for g, w in zip(gradients, weights):
		# assign_sub is the equivalent of -= for TensorFlow variables.
		# Kinda like applying gradient descent, we are going in the direction opposite of the gradient. 
		# This function modifies each of the weights and modifies it.
		w.assign_sub(w * learning_rate)
# %%
def fit(model, images, labels, epochs, batch_size=128):
	# Now we define the training loop.
	# We pass in the model that we are using, the images and labels, then we define the epochs and batch_size simlar to model.fit in Keras.
	for epoch_counter in range(epochs):
		print('Epoch %d' % epoch_counter)

		# Create the iterator object of batches
		batch_generator = BatchGenerator(images, labels, batch_size)
		for batch_counter in range(len(images) // batch_size):
			# Any overflow of an incomplete batch is not used.

			# Get batches of images and labels, calling next() to get the next set. I guess this is like an iterator without actually being an iterator object.
			images_batch, labels_batch = batch_generator.next()

			# Calculate loss for the batch
			loss = one_training_step(model, images_batch, labels_batch)

			# Report the loss every 100 batches.
			if batch_counter % 100 == 0:
				print('loss at batch %d: %.2f' % (batch_counter, loss))

# %%
# Running the fit with 10 epochs
fit(model, train_images, train_labels, epochs=10, batch_size=128)
# We minimize our loss after about 3 epochs. 

# We can evaluate the model by taking the argmax of its predictions over the test images, and comparing it to the expected labels:
predictions = model(test_images)
# Calling .numpy() on a TensorFlow tensor converts it to a NumPy tensor.
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
# I think something is wrong, or the model is bad, because 10% accuracy is about as good as randomly guessing. 
print('accuracy: %.2f' % matches.mean())

# %%
# We can compare to our Keras model from above.
predictions_keras = model_keras(test_images)
predictions_keras = predictions_keras.numpy()
# Create a list of the position of the maximum element
predicted_labels = np.argmax(predictions_keras, axis=1)
matches = predicted_labels == test_labels
print('accuracy: %.2f' % matches.mean())
# %%
