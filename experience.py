import numpy as np
import cupy as cp

from settings import *


class ReplayMemory:
	def __init__(self, capacity=10000):
		self.capacity = capacity
		self.current_state = cp.zeros((self.capacity, NFRAMES, HEIGHT, WIDTH), dtype=cp.uint8)
		self.action = np.zeros(self.capacity, dtype=np.uint8)
		self.reward = cp.zeros(self.capacity, dtype=cp.float32)
		self.next_state = cp.zeros_like(self.current_state)
		self.done = cp.zeros(self.capacity, dtype=cp.bool)
		self.idx = 0
		self.len = 0


	def store_transition(self, cur_state, action, reward, nxt_state, done):
		self.current_state[self.idx] = cp.asarray(cur_state)
		self.action[self.idx] = action
		self.reward[self.idx] = cp.asarray(reward)
		self.next_state[self.idx] = cp.asarray(nxt_state)
		self.done[self.idx] = cp.asarray(done)

		self.idx = (self.idx + 1) % self.capacity
		if self.len < self.capacity:
			self.len+= 1


	def sample_random(self, batch_size):
		i = np.random.choice(self.len, batch_size)
		return self.current_state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]
