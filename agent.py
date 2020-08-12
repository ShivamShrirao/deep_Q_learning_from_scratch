import sys
sys.path.append("../dnn_from_scratch")

from nnet_gpu.network import Sequential
from nnet_gpu.layers import Conv2D,MaxPool,Flatten,Dense,Dropout,BatchNormalization,GlobalAveragePool
from nnet_gpu import optimizers
from nnet_gpu import functions
import numpy as np
import cupy as cp

from settings import *

## Tanh was changed, overlap and past reward was changed, try overlap of NFRAMES - 1 or - 2


def get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=3):
	model=Sequential()
	model.add(Conv2D(num_kernels=32, kernel_size=3, stride=(2, 2), activation=functions.relu, input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(Conv2D(num_kernels=64, kernel_size=3, stride=(2, 2), activation=functions.relu))
	model.add(BatchNormalization())
	model.add(Conv2D(num_kernels=128, kernel_size=3, stride=(2, 2), activation=functions.relu))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(256, activation=functions.relu))
	model.add(Dense(no_of_actions, activation=functions.echo))

	model.compile(optimizer=optimizers.adam, loss=functions.mean_squared_error, learning_rate=0.00005)
	return model


def state_to_gpu(state):
	return cp.asarray(state, dtype=cp.float32)/127.5 - 1

class Agent:
	def __init__(self, actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=1e-5):
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.eps_decay = eps_decay
		self.actions = actions
		self.model = get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=len(self.actions))
		self.target = get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=len(self.actions))
		self.model.summary()
		self.update_target()

	def predict(self, state):
		state = state_to_gpu(state)
		state = cp.expand_dims(state, axis=0)
		return self.model.predict(state)

	def get_action(self, state):
		if self.epsilon > self.min_epsilon:
			self.epsilon-= self.eps_decay

		if np.random.uniform() <= self.epsilon: # random action with epsilon greedy
			return np.random.choice(self.actions)
		else:
			out = self.predict(state)
			return self.actions[cp.argmax(out[0]).item()]

	def train(self, D_exp, batch_size=BATCH_SIZE, gamma=0.95):
		curr_state, action_idxs, rewards, next_state, not_done = D_exp.sample_random(batch_size)
		action_idxs_gpu = cp.asarray(action_idxs)
		curr_gpu = state_to_gpu(curr_state)
		Q_curr   = self.model.forward(curr_gpu)							# predict reward for current state
		
		Qar_next = self.target.predict(state_to_gpu(next_state))		# predict reward for next state
		Qr_next  = Qar_next.max(axis=1)									# get max rewards (greedy)
		Qr_next  = Qr_next * cp.asarray(not_done, dtype=cp.float32)		# zero out next rewards for terminal
		Y_argm   = cp.asarray(rewards, dtype=cp.float32) + gamma*Qr_next

		Y_t = cp.copy(Q_curr)
		Y_t[cp.arange(len(curr_state)), action_idxs_gpu] = Y_argm

		grads = self.model.del_loss(Q_curr, Y_t)
		self.model.backprop(grads)
		self.model.optimizer(self.model.sequence, self.model.learning_rate, self.model.beta)
