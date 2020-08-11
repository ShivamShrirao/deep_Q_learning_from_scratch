import gym
import time
import cv2
import numpy as np

from settings import *
from agent import *
from experience import *

env = gym.make('Pong-v0')

agt = Agent(actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=9e-5)
D_exp = ReplayMemory(capacity=10000)


def preprocess_observation(observation):
	state = observation[34:194] # 160,160,3
	state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
	state = cv2.resize(state, (HEIGHT,WIDTH), interpolation = cv2.INTER_NEAREST)
	return state


def train(D_exp, agt, gamma=1):
	# if D_exp.len > BATCH_SIZE
	curr_state, actions, rewards, next_state, had_done = D_exp.sample_random(min(D_exp.len, BATCH_SIZE))
	curr_gpu = state_to_gpu(curr_state)
	Qar = agt.model.predict(curr_gpu)							# predict reward for current state
	
	Qar_next = agt.model.predict(state_to_gpu(next_state))		# predict reward for next state
	Qr_next  = Qar_next.max(axis=1)								# get max rewards (greedy)
	Qr_next  = Qr_next * cp.asarray(had_done)					# zero out next rewards for terminal
	Y_argm = cp.asarray(rewards) + gamma*Qr_next

	Qar[np.arange(len(curr_state)), actions] = Y_argm
	agt.model.train_on_batch(curr_gpu, Qar)


for i_episode in range(2):
	stacked_state = np.zeros((NFRAMES,HEIGHT,WIDTH), dtype=np.uint8)
	stacked_next = np.zeros_like(stacked_state)
	observation = env.reset()
	state = preprocess_observation(observation)
	stacked_next[:3] = stacked_next[1:]
	stacked_next[3] = state
	ep_score = 0
	for t in range(10000):
		env.render()
		stacked_state[:] = stacked_next			# copy values of now current state
		action = agt.get_action(stacked_state)
		next_observation, reward, done, info = env.step(action)
		ep_score += reward
		observation = next_observation

		state = preprocess_observation(observation)
		stacked_next[:3] = stacked_next[1:]
		stacked_next[3] = state

		D_exp.store_transition(stacked_state, action, reward, stacked_next, done)

		train(D_exp, agt)	
		

		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
	print("Episode score:", ep_score)

env.close()