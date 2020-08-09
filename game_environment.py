import gym
import time
import cv2

env = gym.make('Pong-v0')

for i_episode in range(2):
	observation = env.reset()
	t = 0
	start = time.time()
	while True:
		t+=1
		# env.render()
		cv2.imshow("oo", observation[34:194]) # 160,160,3
		key = cv2.waitKey(1) & 0xff
		if key == ord('q'):
			break
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
	print("Time:", time.time() - start)

env.close()