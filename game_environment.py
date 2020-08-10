import gym
import time
import cv2
# from agent import Agent

env = gym.make('Pong-v0')

for i_episode in range(2):
	observation = env.reset()
	ep_score = 0
	for t in range(10000):
		# env.render()
		state = observation[34:194] # 160,160,3
		action = env.action_space.sample()
		next_observation, reward, done, info = env.step(action)
		ep_score += reward
		observation = next_observation
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
	print("Episode score:", ep_score)

env.close()