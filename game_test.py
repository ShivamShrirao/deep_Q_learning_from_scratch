import gym
import numpy as np
import time
from gym.utils.play import play
import matplotlib.pyplot as plt

env = gym.make('Breakout-v0')
print(env.unwrapped.get_action_meanings())

# play(env)

fps = 60
info = {}

for i_episode in range(1):
    observation = env.reset()
    lives = 5
    fire = True
    ep_score = 0
    start = time.time()
    t = -1
    while 1:
        t+=1
        env.render()
        if fire:
            action = 1
            fire = False
        else:
            action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)
        if lives != info['ale.lives']:
            lives = info['ale.lives']
            reward = -1
            fire = True
        ep_score += reward
        observation = next_observation
        state = observation[34:194:2,::2]#[32:197:2,8:152:2]
        print(state.shape)
        plt.imshow(state)
        plt.show()
        time.sleep(1/fps)
        print('\r', t, action, ep_score, reward, info['ale.lives'], end='  ')
        if done:
            break
    print(f"\rEpisode {i_episode+1} finished after {t+1} timesteps, Score: {ep_score}, Time: {time.time()-start:.2f}")