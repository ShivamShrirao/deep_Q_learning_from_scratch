import gym
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import deque

from settings import *
from agent import *

fps = 144

agt = Agent(actions=[0,2,3], epsilon=0)
agt.model.load_weights("breakout.w8s")

env = gym.make('Breakout-v0').env

# env = wrappers.Monitor(env, '/content/videos/' + str(time.time()) + '/')
for i_episode in range(3):
    obinit = env.reset()
    if not i_episode:
        observation = obinit
        state = preproc_obsv(observation)
        state_que = deque([], maxlen=NFRAMES)
        for i in range(NFRAMES):
            state_que.append(state)
    ep_score = 0
    lives = 5
    fire = True
    preds = []
    reward_history = []
    start = time.time()
    t=-1
    while 1:
        t+=1
        s_s = time.time()
        env.render()
        state = preproc_obsv(observation)
        state_que.append(state)
        if fire:
            action = 1
            fire = False
        else:
            out = agt.predict(state_que)
            pidx = cp.argmax(out[0]).item()
            preds.append(out[0][pidx].item())
            action = agt.actions[pidx]
        next_observation, reward, done, info = env.step(action)
        if lives != info['ale.lives']:
            lives = info['ale.lives']
            reward = -1
            fire = True
        ep_score += reward
        reward_history.append(reward)
        if action==1:
            action = 0
        observation = next_observation
        print('\r', t, ep_score, end='  ')
        if done:
            break
    print(f"\rEpisode {i_episode+1} finished after {t+1} timesteps, Score: {ep_score}, Epsilon: {agt.epsilon:.6f}, Time: {time.time()-start:.2f}")
    # plt.plot(reward_history, label="Reward History")
    # plt.plot(preds, label="Prediction")
    # plt.legend(loc='lower right')
    # plt.show()
env.close()