"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.

Gradients are calculated using backpropagation.

The cost function is the usual squared-error form.
"""


#### Libraries

# Standard Library
import random

# Third-party Libraries
import numpy as np 


class Network(object):

	def __init__(self, sizes):

		"""The list sizes contains the number of neurons in 
		the repsective layers of the network.

		The biases and weights for the network are initialized
		randomly using a Gaussian distribution with mean 0 and 
		variance 1.

		The first layer (the "input" layer) only has weights since
		biases are only ever used in computing the ouputs from 
		later layers.
		"""

		self.num_layers = len(sizes)
		self.sizes = sizes

		# biases are vectors w/ dimensions (y, 1)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		
		# weights are matrices w/ dimeensions (y, x)
		self.weights = [np.random.randn(y, x)
						for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		"""Return the output of the network if 'a' is the input."""

		# compute the vector of activations
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)

		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta,
			test_data=None):

		"""Train the NN using mini-batch stochastic gradient
		descent.  

		The 'training_data' is a list of tuples '(x,y)'
		representing the training inputs and the desired outputs.

		If 'test_data' is provided then the network will be 
		evaluated against the test data after each epoch and 
		partial progress printed out."""

		if test_data: n_test = len(test_data)

		n = len(training_data)

		for j in xrange(epochs):

			# shuffle the training data before batching
			random.shuffle(training_data)

			# break training data into list of mini-batches
			mini_batches = [
				training_data[k: k + mini_batch_size]
				for k in xrange(0, n, mini_batch_size)]

			# Iterate through the mini-batches and update the
			# weights and biases by applying gradient descent
			# using backpropagation
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)

			# print out progress for test data
			if test_data:
				print "Epoch {0}: {1} / {2}".format(
					j, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(j)

	def update_mini_batch(self, mini_batch, eta):

		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini-
		batch.

		The mini-batch is a list of tuples (x, y) and 'eta' is
		the learning rate."""

		# initialize dC/db and dC/dw to vector (matrix) of zeros
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# use backpropogation to update derivatives of C		
		for x, y in mini_batch:
			
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)

			nabla_b = [nb+dnb for nb, dnb 
						in zip(nabla_b, delta_nabla_b)]

			nabla_w = [nw+dnw for nw, dnw 
						in zip(nabla_w, delta_nabla_w)]

		# update the weights using gradient descent
		self.weights = [w - (eta / len(mini_batch)) * nw
						for w, nw in zip(self.weights, nabla_w)]

		# update the biases using gradient descent
		self.biases = [b - (eta / len(mini_batch)) * nb
						for b, nb in zip(self.biases, nabla_b)]


	def backprop(self, x, y):

		"""Return a tuple (nabla_b, nabla_w) representing the
		gradient for the cost function C_x.

		nabla_b and nabla_w are layer-by-layer lists of numpy
		arrays similar to self.biases and self.weights."""

		# initialize the derivative arrays to zeros
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(b.shape) for w in self.weights]

		# activations (needed to compute dC/dw)
		activation = x

		# list to store all the activations, layer by layer
		activations = [x] 

		# list to store all the weighted inputs, layer by layer
		zs = []

		# compute the activations 
		for b, w in zip(self.biases, self.weights):

			# first computing weighted inputs
			z = np.dot(w, activation) + b 

			zs.append(z)

			# then, feed them into sigmoid function 
			activation = sigmoid(z)

			activations.append(activation)

		# backward pass:
		#  compute delta_L (the output error):
		delta = self.cost_derivative(activations[-1], y) * \
			sigmoid_prime(zs[-1])

		#  compute dC/db_L
		nabla_b[-1] = delta

		#  compute dC/dw_L
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		# now, backpropagate it to the previous layers
		for l in xrange(2, self.num_layers):

			z = zs[-l]

			sp = sigmoid_prime(z)

			# notice that delta is updating itself
			delta = np.dot(self.weights[-l+1].transpose(), \
				delta) * sp

			nabla_b[-l] = delta

			nabla_w[-l] = np.dot(delta, \
				activations[-l-1].transpose())

		return (nabla_b, nabla_w)

	def evaluate(self, test_data):

		"""Return the number of test inputs for which the neural
		network outputs the correct result.

		The NN's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation
		""" 		

		test_results = [(np.argmax(self.feedforward(x)), y)
							for (x, y) in test_data]

		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):

		"""Return the vector of partial derivatives dC_x/da
		for the output activations (i.e., the last layer)"""

		return (output_activations - y)

#### Miscellaneous functions
def sigmoid(z):
	"""The sigmoid function"""

	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
	"""Derivative of the sigmoid function"""

	return sigmoid(z) * (1.0 - sigmoid(z))























