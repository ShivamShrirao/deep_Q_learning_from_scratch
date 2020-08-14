import pickle

import matplotlib.pyplot as plt

with open("history.w8s", "rb") as f:
	preds, reward_history = pickle.load(f)

plt.plot(reward_history, label="Reward History")
plt.plot(preds, label="Prediction")
plt.legend(loc='lower right')
plt.show()