import sys
sys.path.append("../dnn_from_scratch")

from nnet_gpu.network import Sequential
from nnet_gpu.layers import Conv2D,MaxPool,Flatten,Dense,Dropout,BatchNormalization,GlobalAveragePool
from nnet_gpu import optimizers
from nnet_gpu import functions
import numpy as np
import cupy as cp


WIDTH = 80
HEIGHT = 80
NFRAMES = 4


def preprocess_observation(observation):
	state = observation[34:194] # 160,160,3
	state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
	state = cv2.resize(state, (HEIGHT,WIDTH), interpolation = cv2.INTER_NEAREST)
	state = np.expand_dims(state, axis=2)
	return state


class ReplayMemory:
	def __init__(self, capacity=10000):
		self.capacity = capacity
		self.current_state = np.zeros((self.capacity, HEIGHT, WIDTH, NFRAMES), dtype=np.uint8)
		self.action = np.zeros(self.capacity, dtype=np.uint8)
		self.reward = np.zeros(self.capacity, dtype=np.uint8)
		self.next_state = np.zeros_like(self.current_state)
		self.done = np.zeros(self.capacity, dtype=np.bool)
		self.idx = 0
		self.len = 0


	def store_transition(self, cur_state, action, reward, nxt_state, done):
		self.current_state[self.idx] = cur_state
		self.action[self.idx] = action
		self.reward[self.idx] = reward
		self.next_state[self.idx] = nxt_state
		self.done[self.idx] = done

		self.idx = (self.idx + 1) % self.capacity
		if self.len < self.capacity:
			self.len+= 1


	def sample_random(batch_size):
		i = np.random.choice(self.len, batch_size)
		return self.current_state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]

class Agent:
	def __init__(self, actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=9e-5):
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.eps_decay = eps_decay
		self.actions = actions
		self.model = self.get_model(input_shape=(WIDTH,HEIGHT,NFRAMES), no_of_actions=len(self.actions))


	def get_action(self, state, epsilon=0.1):
		if np.random.uniform() <= epsilon: # random action with epsilon greedy
			return np.random.choice(self.actions)
		else:
			out = self.model.predict(state)
			return actions[np.argmax(out[0])]

	@staticmethod
	def get_model(input_shape=(WIDTH,HEIGHT,NFRAMES), no_of_actions=3):
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
