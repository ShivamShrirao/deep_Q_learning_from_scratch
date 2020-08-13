import numpy as np
import cupy as cp
from collections import deque

from settings import *

# TODO - Try single memory for current and next states.

class ReplayMemory:
	def __init__(self, capacity=100000):
		self.capacity = capacity
		self.current_state = deque([], maxlen=self.capacity)
		self.action_idx = deque([], maxlen=self.capacity)
		self.reward = deque([], maxlen=self.capacity)
		self.next_state = deque([], maxlen=self.capacity)
		self.ndone = deque([], maxlen=self.capacity)


	def store_transition(self, cur_state, action_idx, reward, nxt_state, done):
		self.current_state.append(cur_state)
		self.action_idx.append(action_idx)
		self.reward.append(reward)
		self.next_state.append(nxt_state)
		self.ndone.append(not done)

	def sample_random(self, batch_size=BATCH_SIZE):
		idxs = np.random.choice(len(self.ndone), batch_size, replace=False)
		cur_state=[]
		action_idx=[]
		reward=[]
		nxt_state=[]
		ndone=[]
		for j in idxs:
			cur_state.append(self.current_state[j])
			action_idx.append(self.action_idx[j])
			reward.append(self.reward[j])
			nxt_state.append(self.next_state[j])
			ndone.append(self.ndone[j])
		return cur_state, action_idx, reward, nxt_state, ndone