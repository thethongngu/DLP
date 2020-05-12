import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
	""" Sigmoid function.
	This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
	"""
	return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
	""" First derivative of Sigmoid function.
	The input to this function should be the value that output from sigmoid function.
	"""
	return y * (1 - y)


class GenData:
	@staticmethod
	def _gen_linear(n=100):
		""" Data generation (Linear)

		Args:
			n (int):    the number of data points generated in total.

		Returns:
			data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
				a data point in 2d space.
			labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
				Each row represents a corresponding label (0 or 1).
		"""
		data = np.random.uniform(0, 1, (n, 2))

		inputs = []
		labels = []

		for point in data:
			inputs.append([point[0], point[1]])

			if point[0] > point[1]:
				labels.append(0)
			else:
				labels.append(1)

		return np.array(inputs), np.array(labels).reshape((-1, 1))

	@staticmethod
	def _gen_xor(n=100):
		""" Data generation (XOR)

		Args:
			n (int):    the number of data points generated in total.

		Returns:
			data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
				a data point in 2d space.
			labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
				Each row represents a corresponding label (0 or 1).
		"""
		data_x = np.linspace(0, 1, n // 2)

		inputs = []
		labels = []

		for x in data_x:
			inputs.append([x, x])
			labels.append(0)

			if x == 1 - x:
				continue

			inputs.append([x, 1 - x])
			labels.append(1)

		return np.array(inputs), np.array(labels).reshape((-1, 1))

	@staticmethod
	def fetch_data(mode, n):
		""" Data gather interface

		Args:
			mode (str): 'Linear' or 'XOR', indicate which generator is used.
			n (int):    the number of data points generated in total.
		"""
		assert mode == 'Linear' or mode == 'XOR'

		data_gen_func = {
			'Linear': GenData._gen_linear,
			'XOR': GenData._gen_xor
		}[mode]

		return data_gen_func(n)


class SimpleNet:
	def __init__(self, hidden_size, num_step, learning_rate, batch_size, print_interval=100):
		""" A hand-crafted implementation of simple network.

		Args:
			hidden_size:    the number of hidden neurons used in this model.
			num_step (optional):    the total number of training steps.
			print_interval (optional):  the number of steps between each reported number.
		"""
		assert batch_size > 0

		self.num_step = num_step
		self.print_interval = print_interval
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.log = []

		# Model parameters initialization
		# Please initiate your network parameters here.
		self.w1 = np.random.rand(hidden_size, 2)
		self.w2 = np.random.rand(hidden_size, hidden_size)
		self.w3 = np.random.rand(1, hidden_size)

	@staticmethod
	def plot_result(data, gt_y, pred_y):
		""" Data visualization with ground truth and predicted data comparison. There are two plots
		for them and each of them use different colors to differentiate the data with different labels.

		Args:
			data:   the input data
			gt_y:   ground truth to the data
			pred_y: predicted results to the data
		"""
		assert data.shape[0] == gt_y.shape[0]
		assert data.shape[0] == pred_y.shape[0]

		plt.figure()

		plt.subplot(1, 2, 1)
		plt.title('Ground Truth', fontsize=18)

		for idx in range(data.shape[0]):
			if gt_y[idx] == 0:
				plt.plot(data[idx][0], data[idx][1], 'ro')
			else:
				plt.plot(data[idx][0], data[idx][1], 'bo')

		plt.subplot(1, 2, 2)
		plt.title('Prediction', fontsize=18)

		for idx in range(data.shape[0]):
			if pred_y[idx] == 0:
				plt.plot(data[idx][0], data[idx][1], 'ro')
			else:
				plt.plot(data[idx][0], data[idx][1], 'bo')

	def forward(self, inputs):
		""" Implementation of the forward pass.
		It should accepts the inputs and passing them through the network and return results.
		"""
		self.a0 = inputs.T

		self.z1 = self.w1 @ self.a0
		self.a1 = sigmoid(self.z1)

		self.z2 = self.w2 @ self.a1
		self.a2 = sigmoid(self.z2)

		self.z3 = self.w3 @ self.a2
		self.a3 = sigmoid(self.z3)

		return self.a3

	def backward(self):
		""" Implementation of the backward pass.
		It should utilize the saved loss to compute gradients and update the network all the way to the front.
		"""
		dldz3 = 2 * self.grad_loss * der_sigmoid(self.a3)
		dldw3 = self.a2 @ dldz3.T

		dldz2 = (self.w3.T @ dldz3) * der_sigmoid(self.a2)
		dldw2 = self.a1 @ dldz2.T

		dldz1 = (self.w2.T @ dldz2) * der_sigmoid(self.a1)
		dldw1 = self.a0 @ dldz1.T

		self.w3 -= self.learning_rate * dldw3.T
		self.w2 -= self.learning_rate * dldw2.T
		self.w1 -= self.learning_rate * dldw1.T

	def train(self, inputs, labels):
		""" The training routine that runs and update the model.

		Args:
			inputs: the training (and testing) data used in the model.
			labels: the ground truth of correspond to input data.
		"""
		# make sure that the amount of data and label is match
		assert inputs.shape[0] == labels.shape[0]

		n = inputs.shape[0]

		for epochs in range(1, self.num_step + 1):
			for idx in range(0, n, self.batch_size):
				# operation in each training step:
				#   1. forward passing
				#   2. compute loss
				#   3. propagate gradient backward to the front
				self.output = self.forward(inputs[idx: idx + self.batch_size, :])
				self.grad_loss = (self.output - labels[idx: idx + self.batch_size, :].T) / self.batch_size
				self.backward()

			if epochs % self.print_interval == 0:
				print('Epochs {}: '.format(epochs), end='')
				error = self.test(inputs, labels, False)
				self.log.append((epochs, error))

		return self.log

	def test(self, inputs, labels, show_info):
		""" The testing routine that run forward pass and report the accuracy.

		Args:
			inputs: the testing data. One or several data samples are both okay.
				The shape is expected to be [BatchSize, 2].
			labels: the ground truth correspond to the inputs.
		"""
		n = inputs.shape[0]

		error = 0.0
		for idx in range(n):
			result = self.forward(inputs[idx:idx+1, :])
			error += np.squeeze(abs(result - labels[idx:idx+1, :]))

		error /= n
		print('loss: %.15f' % error)

		if show_info:
			result = self.forward(inputs)			
			print("Prediction: ")
			for var in np.squeeze(result):
				print('%.15f' % var)
			print()

		return error


def run_experiment(hidden_size, learning_rate, training_step, data_mode, batch_size):

	net = SimpleNet(hidden_size, training_step, learning_rate, batch_size, 100)

	""" Training """
	train_data, train_label = GenData.fetch_data(data_mode, 70)
	training_log = net.train(train_data, train_label)

	""" Testing """
	test_data, test_label = GenData.fetch_data(data_mode, 30)
	print('Testing: ', end='')
	net.test(test_data, test_label, True)

	""" Plotting """
	pred_result = np.round(net.forward(train_data))
	SimpleNet.plot_result(train_data, train_label, pred_result.T)
	plt.savefig('%d_%s_%d_%s_%d_0train.png' % (hidden_size, str(learning_rate), training_step, data_mode, batch_size))

	pred_result = np.round(net.forward(test_data))
	SimpleNet.plot_result(test_data, test_label, pred_result.T)
	plt.savefig('%d_%s_%d_%s_%d_1test.png' % (hidden_size, str(learning_rate), training_step, data_mode, batch_size))

	return training_log

def plot_summary(configs, legends):

	plt.figure()
	for j in range(len(configs)):
		config = configs[j]
		xx, yy = [], []
		
		for i in range(len(config)):
			xx.append(config[i][0])
			yy.append(config[i][1])
		plt.plot(xx, yy, label=legends[j])

	plt.xlabel('epochs')
	plt.ylabel('error')
	plt.legend()
	plt.savefig('summary.png')

if __name__ == '__main__':

	d1 = run_experiment(hidden_size=10, learning_rate=0.05, training_step=1000, data_mode='Linear', batch_size=1)

	# plot_summary([d1, d2, d5], ['1', '10', '70'])
