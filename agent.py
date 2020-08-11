import sys
sys.path.append("../dnn_from_scratch")

from nnet_gpu.network import Sequential
from nnet_gpu.layers import Conv2D,MaxPool,Flatten,Dense,Dropout,BatchNormalization,GlobalAveragePool
from nnet_gpu import optimizers
from nnet_gpu import functions
import numpy as np
import cupy as cp

from settings import *


def get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=3):
	model=Sequential()
	model.add(Conv2D(num_kernels=32, kernel_size=3, stride=(2, 2), activation=functions.relu, input_shape=input_shape))
	model.add(Dropout(0.1))
	model.add(Conv2D(num_kernels=64, kernel_size=3, stride=(2, 2), activation=functions.relu))
	model.add(Dropout(0.2))
	model.add(Conv2D(num_kernels=128, kernel_size=3, stride=(2, 2), activation=functions.relu))
	model.add(Flatten())
	model.add(Dropout(0.3))
	model.add(Dense(256, activation=functions.relu))
	model.add(Dense(no_of_actions, activation=functions.echo))

	model.compile(optimizer=optimizers.adam, loss=functions.mean_squared_error, learning_rate=0.01)
	return model


def state_to_gpu(state):
	state = state.transpose(0,2,3,1)
	state_gpu = cp.asarray(state).astype(cp.float32)
	return state_gpu


class Agent:
	def __init__(self, actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=9e-5):
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.eps_decay = eps_decay
		self.actions = actions
		self.model = get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=len(self.actions))
		self.model.summary()


	def get_action(self, state):
		if self.epsilon > self.min_epsilon:
			self.epsilon-= self.eps_decay

		if np.random.uniform() <= self.epsilon: # random action with epsilon greedy
			return np.random.choice(self.actions)
		else:
			state = np.expand_dims(state, axis=0)
			state = state_to_gpu(state)
			out = self.model.predict(state)
			return self.actions[np.argmax(out[0].get())]

	