import numpy as np
import cupy as cp
from collections import deque
from numpy.random import default_rng

from settings import *

# TODO - Try single memory for current and next states.

class ReplayMemory:
	def __init__(self, capacity=1_000_000, nlap=1):
		self.capacity = capacity
		self.current_state = np.zeros((capacity,HEIGHT,WIDTH), dtype=np.uint8)
		self.action_idx = np.zeros(capacity, dtype=np.int8)
		self.reward = np.zeros(capacity, dtype=np.int8)
		self.ndone = np.zeros(capacity, dtype=np.bool)
		self.idx = 0
		self.rng = default_rng()
		self.nlap = nlap
		self.batch_size = BATCH_SIZE
		self.min_idx = NFRAMES - 1
		self.idx_len = NFRAMES + self.nlap
		self.lens = self.get_lens(self.batch_size)
		self.len = 0

	def store_transition(self, cur_state, action_idx, reward, done):
		self.current_state[self.idx] = cur_state
		self.action_idx[self.idx] = action_idx
		self.reward[self.idx] = reward
		self.ndone[self.idx] = not done
		self.idx = (self.idx + 1) % self.capacity
		if self.len < self.capacity:
			self.len += 1

	def get_lens(self, batch_size):
		lens = np.full(batch_size, self.idx_len)
		np.cumsum(lens, out=lens)
		return lens

	def indices(self, start, end, batch_size):
		if batch_size != self.batch_size:
			self.batch_size = batch_size
			self.lens = self.get_lens(self.batch_size)
		i = np.ones(self.lens[-1], dtype=int)
		i[0] = start[0]
		i[self.lens[:-1]] += start[1:]
		i[self.lens[:-1]] -= end[:-1]
		np.cumsum(i, out=i)
		return i.reshape(batch_size, self.idx_len)

	def sample_random(self, batch_size=BATCH_SIZE):
		idxs = self.rng.choice(np.arange(self.min_idx, self.len - self.nlap), size=batch_size, replace=False)
		action_idx = self.action_idx[idxs]
		reward = self.reward[idxs]
		ndone = self.ndone[idxs]

		state_idxs = self.indices(idxs-self.min_idx, idxs+self.nlap, batch_size)
		states = self.current_state[state_idxs]
		cur_state = np.moveaxis(states[:,:NFRAMES], 1, -1)
		nxt_state = np.moveaxis(states[:,self.nlap:], 1, -1)

		return cur_state, action_idx, reward, nxt_state, ndone