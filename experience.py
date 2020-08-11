import numpy as np
import cupy as cp
from collections import deque

from settings import *


class ReplayMemory:
	def __init__(self, capacity=100000):
		self.capacity = capacity
		self.current_state = deque([], maxlen=self.capacity)
		self.action = deque([], maxlen=self.capacity)
		self.reward = deque([], maxlen=self.capacity)
		self.next_state = deque([], maxlen=self.capacity)
		self.done = deque([], maxlen=self.capacity)


	def store_transition(self, cur_state, action, reward, nxt_state, done):
		self.current_state.append(cur_state)
		self.action.append(action)
		self.reward.append(reward)
		self.next_state.append(nxt_state)
		self.done.append(done)

	def sample_random(self, batch_size=BATCH_SIZE):
		idxs = np.random.choice(len(self.done), batch_size, replace=False)
		cur_state=[]
		action=[]
		reward=[]
		nxt_state=[]
		done=[]
		for j in idxs:
			cur_state.append(self.current_state[j])
			action.append(self.action[j])
			reward.append(self.reward[j])
			nxt_state.append(self.next_state[j])
			done.append(self.done[j])
		return cur_state, action, reward, nxt_state, done