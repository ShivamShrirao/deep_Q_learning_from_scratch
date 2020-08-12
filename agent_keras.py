import tensorflow as tf
import numpy as np

from settings import *


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
		self.model.summary()


	def get_action(self, state):
		if self.epsilon > self.min_epsilon:
			self.epsilon-= self.eps_decay

		if np.random.uniform() <= self.epsilon: # random action with epsilon greedy
			return np.random.choice(self.actions)
		else:
			state = state_to_gpu(state)
			state = np.expand_dims(state, axis=0)
			out = self.model.predict(state)
			return self.actions[np.argmax(out[0])]

	def train(self, D_exp, batch_size=BATCH_SIZE, gamma=0.99):
		curr_state, action_idxs, rewards, next_state, not_done = D_exp.sample_random(BATCH_SIZE)
		curr_gpu = state_to_gpu(curr_state)
		# Qar = self.model.predict(curr_gpu)							# predict reward for current state
		
		Qar_next = self.model.predict(state_to_gpu(next_state))		# predict reward for next state
		Qr_next  = Qar_next.max(axis=1)								# get max rewards (greedy)
		Qr_next  = Qr_next * np.asarray(not_done)					# zero out next rewards for terminal
		Y_argm   = np.asarray(rewards) + gamma*Qr_next

		Qar = np.zeros_like(Qar_next)
		Qar[np.arange(len(curr_state)), action_idxs] = Y_argm
		self.model.train_on_batch(curr_gpu, Qar)

	