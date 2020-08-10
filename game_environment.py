import gym
import time
import cv2
import numpy as np
from agent import *

env = gym.make('Pong-v0')

D_exp = ReplayMemory(capacity=10000)

for i_episode in range(2):
	stacked_state = np.zeros((NFRAMES,HEIGHT,WIDTH,1), dtype=np.uint8)
	stacked_next = np.zeros_like(stacked_state)
	observation = env.reset()
	state = preprocess_observation(observation)
	stacked_next[:3] = stacked_next[1:]
	stacked_next[3] = state
	ep_score = 0
	for t in range(10000):
		# env.render()
		stacked_state[:] = stacked_next			# copy values of now current state
		action = env.action_space.sample()
		next_observation, reward, done, info = env.step(action)
		ep_score += reward
		observation = next_observation

		state = preprocess_observation(observation)
		stacked_next[:3] = stacked_next[1:]
		stacked_next[3] = state

		D_exp.store_transition(stacked_state, action, reward, stacked_next, done)

		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
	print("Episode score:", ep_score)

env.close()