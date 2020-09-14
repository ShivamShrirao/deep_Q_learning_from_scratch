from nnet_gpu.stream_handler import stream_maps
import numpy as np
import cupy as cp
from copy import deepcopy
import cv2

from settings import *

# TODO - https://arxiv.org/pdf/1710.02298.pdf

def state_to_gpu(state):
	return cp.asarray(state, dtype=cp.float32)/127.5 - 1

def sample_to_gpu(curr_state, action_idxs, rewards, next_state, not_done):
	curr_gpu 		= state_to_gpu(curr_state)
	action_idxs_gpu = cp.asarray(action_idxs)
	rewards_gpu 	= cp.asarray(rewards, dtype=cp.float32)
	next_gpu 		= state_to_gpu(next_state)
	not_done_gpu 	= cp.asarray(not_done, dtype=cp.float32)
	return curr_gpu, action_idxs_gpu, rewards_gpu, next_gpu, not_done_gpu


class BaseAgent:
	def __init__(self, actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=2e-6, target_update_thresh=1000, grad_clip=True, continue_decay=False):
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon                          # minimum value for epsilin
		self.eps_decay = eps_decay                              # decay rate for epsilon
		self.actions = actions 									# actions in an array
		self.grad_clip = grad_clip								# whether to clip gradients
		self.continue_decay = continue_decay
		self.target_update_counter = 0
		self.target_update_thresh = target_update_thresh        # interval of updating target model
		self.stream = stream_maps.get_next_stream()

	def predict(self, state_que):								# input the observations(state_que) of length NFRAMES
		state = state_to_gpu(state_que)
		state = cp.moveaxis(state, 0, -1)
		state = cp.expand_dims(state, axis=0)
		return self.model.predict(state)						# predict output


	def get_action(self, state_que):
		if self.epsilon > self.min_epsilon:
			self.epsilon-= self.eps_decay						# decay the epsilon
		else:
			if self.continue_decay:
				self.min_epsilon/=10
				self.eps_decay/=10

		if np.random.uniform() <= self.epsilon:					# random action with epsilon greedy
			return np.random.choice(self.actions)
		else:
			out = self.predict(state_que)						# Else model predicts action
			return self.actions[cp.argmax(out[0]).item()]


	def train(self, D_exp, batch_size=BATCH_SIZE, gamma=0.99):
		curr_state, action_idxs, rewards, next_state, not_done = sample_to_gpu(*D_exp.sample_random(batch_size))
		irange   = cp.arange(batch_size)						# index range

		Q_curr   = self.model.forward(curr_state, training=True)# predict reward for current state

		Qtr_next = self.get_Qtr_next(next_state, irange)		# Get Q target value for next state
		Y_argm   = rewards + gamma*not_done*Qtr_next

		Y_t = cp.copy(Q_curr)
		Y_t[irange, action_idxs] = Y_argm						# set the target values for actions taken

		grads = self.model.del_loss(Q_curr, Y_t)				# calculate gradients
		if self.grad_clip:
			grads = grads.clip(-1, 1)
		self.model.backprop(grads)								# backprop
		self.model.optimizer(self.model.sequence, self.model.learning_rate, self.model.beta)	# update weights
		self.target_update_counter+=1
		if self.target_update_counter > self.target_update_thresh:
			self.update_target()								# update target network weights
			self.target_update_counter = 0
		return grads


	def update_target(self):
		with self.stream:
			self.target.weights = deepcopy(self.model.weights)


# https://arxiv.org/pdf/1312.5602.pdf
class DQN_Agent(BaseAgent):
	def __init__(self, actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=2e-6, target_update_thresh=1000, grad_clip=True, continue_decay=True):
		super().__init__(actions=actions, epsilon=epsilon, min_epsilon=min_epsilon, eps_decay=eps_decay, target_update_thresh=target_update_thresh, grad_clip=grad_clip, continue_decay=continue_decay)

	def get_Qtr_next(self, next_state, irange):
		Qt_next  = self.target.predict(next_state)				# predict reward for next state
		Qtr_next = Qt_next.max(axis=1)							# get max rewards (greedy)
		return Qtr_next


# https://arxiv.org/pdf/1509.06461.pdf
class DDQN_Agent(BaseAgent):
	def __init__(self, actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=2e-6, target_update_thresh=1000, grad_clip=True, continue_decay=True):
		super().__init__(actions=actions, epsilon=epsilon, min_epsilon=min_epsilon, eps_decay=eps_decay, target_update_thresh=target_update_thresh, grad_clip=grad_clip, continue_decay=continue_decay)

	def get_Qtr_next(self, next_state, irange):
		Q_next   = self.model.predict(next_state)				# for actions of next state
		Qt_next  = self.target.predict(next_state)				# predict reward for next state
		Qtr_next = Qt_next[irange, Q_next.argmax(axis=1)]		# select by actions given by model(online network)
		return Qtr_next
