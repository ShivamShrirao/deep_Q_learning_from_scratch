import sys
sys.path.append("../dnn_from_scratch")

from nnet_gpu.network import Sequential
from nnet_gpu.layers import Conv2D,MaxPool,Flatten,Dense,Dropout,BatchNormalization,GlobalAveragePool
from nnet_gpu import optimizers
from nnet_gpu import functions
import numpy as np
import cupy as cp

class ReplayMemory:
	def __init__(self, capacity=10000)


class Agent:
	def __init__(self, actions=[0,1], epsilon=1, min_epsilon=0.1):
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.actions = actions
		self.model = self.get_model(no_of_actions=len(self.actions))


	def preprocess(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (80,80), interpolation = cv2.INTER_NEAREST)
		return img

	def get_action(self, state, epsilon=0.1):
		if np.random.uniform() <= epsilon: # random action with epsilon greedy
			return np.random.choice(self.actions)
		else:
			out = self.model.predict(state)
			return actions[np.argmax(out[0])]

	@staticmethod
	def get_model(input_shape=(80,80,4), no_of_actions=4):
		model=Sequential()
		model.add(Conv2D(num_kernels=32, kernel_size=3, stride=(2, 2), activation=functions.relu, input_shape=input_shape))
		model.add(Dropout(0.1))
		model.add(Conv2D(num_kernels=64, kernel_size=3, stride=(2, 2), activation=functions.relu))
		model.add(Dropout(0.2))
		model.add(Conv2D(num_kernels=128, kernel_size=3, stride=(2, 2), activation=functions.relu))
		model.add(Dropout(0.3))
		model.add(Flatten())
		model.add(Dense(256, activation=functions.relu))
		model.add(Dense(no_of_actions, activation=functions.echo))

		model.compile(optimizer=optimizers.adam, loss=functions.mean_squared_error, learning_rate=0.01)
		return model
