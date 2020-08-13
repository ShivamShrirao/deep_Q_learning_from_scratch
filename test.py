import gym
import time
import numpy as np

from settings import *
from agent import *
from experience import *
from atari_wrappers import *

fps = 60

agt = Agent(actions=[0,2,3], epsilon=0)
agt.model.load_weights("model.w8s")

env = gym.make('Pong-v0')
env = FrameStack(env, NFRAMES)      # preprocess and stack frames

for i_episode in range(3):
    observation = env.reset()
    ep_score = 0
    start = time.time()
    for t in range(10000):
        env.render()
        action = agt.get_action(observation)
        next_observation, reward, done, info = env.step(action)
        ep_score += reward
        observation = next_observation
        # time.sleep(1/fps)
        if done:
            break
        print('\r', t, action, ep_score, end='  ')
    print(f"\rEpisode {i_episode+1} finished after {t+1} timesteps, Score: {ep_score}, Epsilon: {agt.epsilon:.6f}, Time: {time.time()-start:.2f}")
env.close()