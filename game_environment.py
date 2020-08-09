import gym
import time

env = gym.make('Pong-v0')

for i_episode in range(20):
	observation = env.reset()
	t = 0
	start = time.time()
	while True:
		t+=1
		env.render()
		# print(observation.shape)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
	print("Time:", time.time() - start)
env.close()