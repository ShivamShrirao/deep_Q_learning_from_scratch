# Deep_Q_learning

Learning to play games using just the visual input.

Implemented it from understanding of [research paper](https://arxiv.org/pdf/1312.5602.pdf) and network is in my own library. https://github.com/ShivamShrirao/dnn_from_scratch .

Using OpenAI gym as game environment.

## Results after overnight training on Colab.
Breakout                        |     Pong
:------------------------------:|:-------------------------------:
![Breakout](/pics/breakout.gif) | ![Pong](/pics/pong.gif)
[![Breakout Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShivamShrirao/deep_Q_learning_from_scratch/blob/master/Breakout_Deep_Q_RL.ipynb) | [![Pong Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShivamShrirao/deep_Q_learning_from_scratch/blob/master/Pong_Deep_Q_RL.ipynb)

The agent has learned the mechanics of the game, formed a few strategies and is able to consistently score good points.

## Actual Reward v/s Q Prediction for breakout
![RewardVprediction](/pics/Figure_breakout.png)

Got inspiration from watching this video https://youtu.be/rFwQDDbYTm4 and read the paper (https://arxiv.org/pdf/1312.5602.pdf).
