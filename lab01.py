import numpy as np
import matplotlib.pyplot as plt

DATA_SIZE = 100
TRAINING_STEP = 100
LEARNING_RATE = 0.01
LAYER_0 = 2
LAYER_1 = 10
LAYER_2 = 10
LAYER_3 = 1

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

def generate_data(mode='linear'):
	x = np.zeros((DATA_SIZE, 2))
	y = np.zeros((DATA_SIZE, 1))

	if mode == 'linear':
		x[:, 0] = np.random.sample(DATA_SIZE)
		x[:, 1] = np.random.sample(DATA_SIZE)
		y = np.array([0 for i in range(DATA_SIZE) if x[i][0] - x[i][1] > 0 else 1])
	if mode == 'xor':
		for i in range(DATA_SIZE / 2):
			x[i] = []

def init_weight():
	w1 = np.random.rand(LAYER_0, LAYER_1)
	w2 = np.random.rand(LAYER_1, LAYER_2)
	w3 = np.random.rand(LAYER_2, LAYER_3)

	return w1, w2, w3

def forward(x, w1, w2, w3):
	z1 = sigmoid(x @ w1)
	z2 = sigmoid(z1 @ w2)
	y_hat = sigmoid(z2 @ w3)

	return y_hat, z1, z2

def get_loss(y, y_hat):
	return y - y_hat

def backward(loss, z1, z2):
	grad_y_hat = 2.0 * loss

	s = sigmoid(z2 @ w3)
	grad_w3 = grad_y_hat @ (s @ (1 - s)) @ z2

	s = sigmoid(z1 @ w2)
	grad_w2 = grad_w3 @ (s @ (1 - s)) @ z1

	s = sigmoid(x @ w1)
	grad_w1 = grad_w2 @ (s @ (1 - s)) @ x

	return grad_w1, grad_w2, grad_w3

def update_weight(w1, w2, w3, grad):
	grad_w1, grad_w2, grad_w3 = grad
	w1 -= LEARNING_RATE * grad_w1
	w2 -= LEARNING_RATE * grad_w2
	w3 -= LEARNING_RATE * grad_w3

	return w1, w2, w3

def visualize(x, prediction, ground_truth):
	fig, ax = plt.subplots(1, 2)

	red_predict, blue_predict = [], []
	for i in range(len(x)):
		red_predict.append(x[i]) if prediction[i] == 0 else blue_predict.append(x[i])

	red_truth, blue_truth = [], []
	for i in range(len(x)):
		red_truth.append(x[i]) if ground_truth[i] == 0 else blue_truth.append(x[i])

	ax[0][0].plot(red_predict[:][0], red_predict[:][1], 'red')
	ax[0][0].plot(blue_predict[:][0], blue_predict[:][1], 'blue')
	ax[0][1].plot(red_truth[:][0], red_truth[:][1], 'red')
	ax[0][1].plot(blue_truth[:][0], blue_truth[:][1], 'blue')

	plt.show()

		
x, y = generate_data()
w1, w2, w3 = init_weight()

for step in range(TRAINING_STEP):
	
	y_hat, z1, z2 = forward(x, w1, w2, w3)
	loss = get_loss(y, y_hat)
	grad = backward(loss, z1, z2)
	w1, w2, w3 = update_weight(w1, w2, w3, grad)

	if step % 1000 == 0:
		print("Step %d loss: %d" % np.sum(np.sqr(loss)))

prediction = forward(x, w1, w2, w3)
ground_truth = y
visualize(x, prediction, ground_truth)
