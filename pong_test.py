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
agt.model.load_weights("pong.w8s")

env = gym.make('Pong-v0').env

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
    preds = []
    reward_history = []
    start = time.time()
    t = -1
    while 1:
        t+=1
        env.render()
        state = preproc_obsv(observation)
        state_que.append(state)
        # action = agt.get_action(state_que)
        out = agt.predict(state_que)
        pidx = cp.argmax(out[0]).item()
        preds.append(out[0][pidx].item())
        action = agt.actions[pidx]
        next_observation, reward, done, info = env.step(action)
        ep_score += reward
        reward_history.append(reward)
        observation = next_observation
        # time.sleep(1/fps)
        if done:
            break
        print('\r', t, action, ep_score, end='  ')
    print(f"\rEpisode {i_episode+1} finished after {t+1} timesteps, Score: {ep_score}, Epsilon: {agt.epsilon:.6f}, Time: {time.time()-start:.2f}")
    plt.plot(reward_history, label="Reward History")
    plt.plot(preds, label="Prediction")
    plt.legend(loc='lower right')
    plt.show()
env.close()