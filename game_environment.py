import gym
import time
import cv2
import numpy as np

from settings import *
from agent import *
from experience import *

env = gym.make('Pong-v0')

agt = Agent(actions=[0,2,3], epsilon=1, min_epsilon=0.1, eps_decay=9e-5)
D_exp = ReplayMemory(capacity=100000)


def preprocess_observation(observation):
	state = observation[34:194:2,::2] # 80,80,3
	state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
	# state = cv2.resize(state, (HEIGHT,WIDTH), interpolation = cv2.INTER_NEAREST)
	return state


for i_episode in range(2):
	stacked_state = np.zeros((NFRAMES,HEIGHT,WIDTH), dtype=np.uint8)
	stacked_next = np.zeros_like(stacked_state)
	observation = env.reset()
	state = preprocess_observation(observation)
	stacked_next[:3] = stacked_next[1:]
	stacked_next[3] = state
	ep_score = 0
	for t in range(10000):
		# env.render()
		start = time.time()
		stacked_state[:] = stacked_next			# copy values of now current state
		action = agt.get_action(stacked_state)
		next_observation, reward, done, info = env.step(action)
		ep_score += reward
		observation = next_observation

		state = preprocess_observation(observation)
		stacked_next[:3] = stacked_next[1:]			# REMOVE OVERLAPPED
		stacked_next[3] = state

		D_exp.store_transition(stacked_state, action, reward, stacked_next, done)

		if D_exp.len > BATCH_SIZE:
			agt.train(D_exp)
		
		if done:
			break
		print('\r', t, time.time()-start, end='  ')
	print(f"\nEpisode finished after {t+1} timesteps, Score: {ep_score}, Epsilon: {agt.epsilon}")
	agt.model.save_weights("model.w8s")

env.close()