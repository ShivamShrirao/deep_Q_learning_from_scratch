import gym
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import deque

from settings import *
from agent import *

from nnet_gpu.network import Sequential
from nnet_gpu.layers import Conv2D,Flatten,Dense,Dropout
from nnet_gpu import optimizers
from nnet_gpu import functions

HEIGHT = 80
WIDTH = 72

fps = 144

def get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=3):
    model=Sequential()
    model.add(Conv2D(num_kernels=32, kernel_size=3, stride=(2, 2), activation=functions.relu, input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(Conv2D(num_kernels=64, kernel_size=3, stride=(2, 2), activation=functions.relu))
    model.add(Dropout(0.2))
    model.add(Conv2D(num_kernels=128, kernel_size=3, stride=(2, 2), activation=functions.relu))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation=functions.relu))
    model.add(Dense(no_of_actions, activation=functions.echo))

    model.compile(optimizer=optimizers.adam, loss=functions.mean_squared_error, learning_rate=0.0001)
    return model

def preproc_obsv(obsv):
    obsv = cv2.cvtColor(obsv, cv2.COLOR_RGB2GRAY)
    obsv = obsv[34:194:2,8:152:2]
    return obsv

agt = DQN_Agent(actions=[0,2,3], epsilon=0)
agt.model = get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=len(agt.actions))
agt.target = get_model(input_shape=(HEIGHT,WIDTH,NFRAMES), no_of_actions=len(agt.actions))
agt.model.load_weights("breakout.w8s")
agt.update_target()
agt.model.summary()

env = gym.make('Breakout-v0')

# env = wrappers.Monitor(env.env, '/content/videos/' + str(time.time()) + '/')
for i_episode in range(10):
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
    fired = False
    preds = [0]
    reward_history = [0]
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
            if fired:
                fire = False
            fired = True
        else:
            out = agt.predict(state_que)
            pidx = cp.argmax(out[0]).item()
            preds.append(out[0][pidx].item())
            action = agt.actions[pidx]
        next_observation, reward, done, info = env.step(action)
        ep_score += reward
        if lives != info['ale.lives']:
            lives = info['ale.lives']
            reward = -2
            fire = True
            fired = False
        reward_history.append(reward)
        if action==1:
            action = 0
        observation = next_observation
        print('\r', t, ep_score, reward, f'{preds[-1]:.2f}', end='  ')
        if done:
            break
    print(f"\rEpisode {i_episode+1} finished after {t+1} timesteps, Score: {ep_score}, Epsilon: {agt.epsilon:.6f}, Time: {time.time()-start:.2f}")
    with open("history.w8s", "wb") as f:
        pickle.dump((preds, reward_history),f)
    # plt.plot(reward_history, label="Reward History")
    # plt.plot(preds, label="Prediction")
    # plt.legend(loc='lower right')
    # plt.show()
env.close()