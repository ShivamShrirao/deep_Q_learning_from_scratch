import tensorflow as tf
import numpy as np

from settings import *

print("Keras isn't functional right now.")

def get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=3):
	model=tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same', activation='relu', input_shape=input_shape))
	# model.add(tf.keras.layers.Dropout(0.1))
	model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'))
	# model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=(2, 2), padding='same', activation='relu'))
	# model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=(2, 2), padding='same', activation='relu'))
	model.add(tf.keras.layers.Flatten())
	# model.add(tf.keras.layers.Dropout(0.3))
	model.add(tf.keras.layers.Dense(256, activation='relu'))
	model.add(tf.keras.layers.Dense(no_of_actions, activation='linear'))

	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00005), loss='mse', metrics=['accuracy'])
	return model


def state_to_gpu(state):
	return np.asarray(state, dtype=np.float32)/127.5 - 1

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
		grads = grads.clip(-1, 1)
		self.model.backprop(grads)
		self.model.optimizer(self.model.sequence, self.model.learning_rate, self.model.beta)
		return grads


	# https://arxiv.org/pdf/1509.06461.pdf
	def DDQN_Qtr_next(self, next_state, irange):
		Q_next   = self.model.predict(next_state)				# for actions of next state
		Qt_next  = self.target.predict(next_state)				# predict reward for next state
		Qtr_next = Qt_next[irange, Q_next.argmax(axis=1)]		# select by actions given by model
		return Qtr_next