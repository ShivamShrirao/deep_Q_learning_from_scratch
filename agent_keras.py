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

	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss='mse', metrics=['accuracy'])
	return model


def preproc_obsv(obsv):
    obsv = cv2.cvtColor(obsv, cv2.COLOR_RGB2GRAY)
    obsv = obsv[34:194:2,::2]
    return obsv

def state_to_gpu(state):
	return np.asarray(state, dtype=np.float32)/127.5 - 1

def sample_to_gpu(curr_state, action_idxs, rewards, next_state, not_done):
	curr_gpu 		= state_to_gpu(curr_state)
	action_idxs_gpu = np.asarray(action_idxs)
	rewards_gpu 	= np.asarray(rewards, dtype=np.float32)
	next_gpu 		= state_to_gpu(next_state)
	not_done_gpu 	= np.asarray(not_done, dtype=np.float32)
	return curr_gpu, action_idxs_gpu, rewards_gpu, next_gpu, not_done_gpu


class Agent:
	def __init__(self, actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=2e-6, target_update_thresh=1000, grad_clip=True):
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.eps_decay = eps_decay
		self.actions = actions
		self.grad_clip = grad_clip
		self.model = get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=len(self.actions))
		self.target = get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=len(self.actions))
		self.target_update_counter = 0
		self.target_update_thresh = target_update_thresh
		self.model.summary()
		self.get_Qtr_next = self.DQN_Qtr_next
		self.stream = stream_maps.get_next_stream()
		self.update_target()

	def predict(self, state_que):
		state = state_to_gpu(state_que)
		state = np.moveaxis(state, 0, -1)
		state = np.expand_dims(state, axis=0)
		return self.model.predict(state)


	def get_action(self, state_que):
		if self.epsilon > self.min_epsilon:
			self.epsilon-= self.eps_decay

		if np.random.uniform() <= self.epsilon:					 # random action with epsilon greedy
			return np.random.choice(self.actions)
		else:
			out = self.predict(state_que)
			return self.actions[np.argmax(out[0]).item()]


	def train(self, D_exp, batch_size=BATCH_SIZE, gamma=0.99):
		curr_state, action_idxs, rewards, next_state, not_done = sample_to_gpu(*D_exp.sample_random(batch_size))
		irange   = np.arange(batch_size)						# index range

		Q_curr   = self.model.forward(curr_state)				# predict reward for current state

		Qtr_next = self.get_Qtr_next(next_state, irange)
		Y_argm   = rewards + gamma*not_done*Qtr_next

		Y_t = np.copy(Q_curr)
		Y_t[irange, action_idxs] = Y_argm

		grads = self.model.del_loss(Q_curr, Y_t)
		if self.grad_clip:
			grads = grads.clip(-1, 1)
		self.model.backprop(grads)
		self.model.optimizer(self.model.sequence, self.model.learning_rate, self.model.beta)
		self.target_update_counter+=1
		if self.target_update_counter > self.target_update_thresh:
			self.update_target()
			self.target_update_counter = 0
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
		with self.stream:
			self.target.weights = deepcopy(self.model.weights)