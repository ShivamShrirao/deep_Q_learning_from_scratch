import sys
sys.path.append("../dnn_from_scratch")

from nnet_gpu.network import Sequential
from nnet_gpu.layers import Conv2D,Flatten,Dense,Dropout
from nnet_gpu import optimizers
from nnet_gpu import functions
import numpy as np
import cupy as cp
import io

from settings import *

## tanh was changed, overlap and past reward was changed, try overlap of NFRAMES - 1 or - 2


def get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=3):
	model=Sequential()
	model.add(Conv2D(num_kernels=32, kernel_size=3, stride=(2, 2), activation=functions.relu, input_shape=input_shape))
	model.add(Conv2D(num_kernels=64, kernel_size=3, stride=(2, 2), activation=functions.relu))
	model.add(Conv2D(num_kernels=128, kernel_size=3, stride=(2, 2), activation=functions.relu))
	model.add(Flatten())
	model.add(Dense(512, activation=functions.relu))
	model.add(Dense(no_of_actions, activation=functions.tanh))

	model.compile(optimizer=optimizers.adam, loss=functions.mean_squared_error, learning_rate=0.00005)
	return model


def state_to_gpu(state):
	return cp.asarray(state, dtype=cp.float32)/127.5 - 1

def sample_to_gpu(curr_state, action_idxs, rewards, next_state, not_done):
	curr_gpu 		= state_to_gpu(curr_state)
	action_idxs_gpu = cp.asarray(action_idxs)
	rewards_gpu 	= cp.asarray(rewards, dtype=cp.float32)
	next_gpu 		= state_to_gpu(next_state)
	not_done_gpu 	= cp.asarray(not_done, dtype=cp.float32)
	return curr_gpu, action_idxs_gpu, rewards_gpu, next_gpu, not_done_gpu


class Agent:
	def __init__(self, actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=1e-6):
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.eps_decay = eps_decay
		self.actions = actions
		self.model = get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=len(self.actions))
		self.target = get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=len(self.actions))
		self.update_target()
		self.model.summary()
		self.get_Qtr_next = self.DDQN_Qtr_next


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
			# print(out)
			return self.actions[cp.argmax(out[0]).item()]


	def train(self, D_exp, batch_size=BATCH_SIZE, gamma=0.99):
		curr_state, action_idxs, rewards, next_state, not_done = sample_to_gpu(*D_exp.sample_random(batch_size))
		irange   = cp.arange(batch_size)						# index range

		Q_curr   = self.model.forward(curr_state)				# predict reward for current state

		Qtr_next = self.get_Qtr_next(next_state, irange)
		Y_argm   = rewards + gamma*not_done*Qtr_next

		Y_t = cp.copy(Q_curr)
		Y_t[irange, action_idxs] = Y_argm

		grads = self.model.del_loss(Q_curr, Y_t)
		# grads = grads.clip(-1, 1)
		self.model.backprop(grads)
		self.model.optimizer(self.model.sequence, self.model.learning_rate, self.model.beta)
		return grads


	# https://arxiv.org/pdf/1509.06461.pdf
	def DDQN_Qtr_next(self, next_state, irange):
		Q_next   = self.model.predict(next_state)				# for actions of next state
		Qt_next  = self.target.predict(next_state)				# predict reward for next state
		Qtr_next = Qt_next[irange, Q_next.argmax(axis=1)]		# select by actions given by model
		return Qtr_next


	# https://arxiv.org/pdf/1312.5602.pdf
	def DQN_Qtr_next(self, next_state, irange):
		Qt_next  = self.target.predict(next_state)				# predict reward for next state
		Qtr_next = Qt_next.max(axis=1)							# get max rewards (greedy)
		return Qtr_next


	def update_target(self):
		f = io.BytesIO()
		self.model.save_weights(f)
		f.seek(0)
		self.target.load_weights(f)
		f.close()
