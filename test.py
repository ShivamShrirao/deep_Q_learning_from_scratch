import gym
import time
import numpy as np
import pickle

from settings import *
from agent import *
from experience import *
from atari_wrappers import *

fps = 144

agt = Agent(actions=[0,2,3], epsilon=0)
agt.model.load_weights("model.w8s")

env = gym.make('Pong-v0')
env = FrameStack(env, NFRAMES)      # preprocess and stack frames

for i_episode in range(3):
    observation = env.reset()
    ep_score = 0
    preds = []
    reward_history = []
    start = time.time()
    t = -1
    while 1:
        t+=1
        env.render()
        # action = agt.get_action(observation)
        out = agt.predict(observation)
        pidx = cp.argmax(out[0]).item()
        preds.append(out[0][pidx].item())
        action = agt.actions[pidx]
        next_observation, reward, done, info = env.step(action)
        ep_score += reward
        reward_history.append(reward)
        observation = next_observation
        time.sleep(1/fps)
        if done:
            break
        print('\r', t, action, ep_score, end='  ')
    print(f"\rEpisode {i_episode+1} finished after {t+1} timesteps, Score: {ep_score}, Epsilon: {agt.epsilon:.6f}, Time: {time.time()-start:.2f}")
    with open("history.w8s","wb") as f:
        pickle.dump([preds, reward_history], f)
env.close()