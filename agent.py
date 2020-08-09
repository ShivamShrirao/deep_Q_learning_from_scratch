import sys
sys.path.append("../dnn_from_scratch")

from nnet_gpu.network import Sequential
from nnet_gpu.layers import Conv2D,MaxPool,Flatten,Dense,Dropout,BatchNormalization,GlobalAveragePool
from nnet_gpu import optimizers
from nnet_gpu import functions
import numpy as np
import cupy as cp


def get_model(input_shape=(32,32,3), no_of_actions=4):
	model=Sequential()
	model.add(Conv2D(num_kernels=32, kernel_size=3, activation=functions.relu, input_shape=input_shape))
	model.add(Conv2D(num_kernels=32, kernel_size=3, stride=(2, 2), activation=functions.relu))
	model.add(BatchNormalization())
	# model.add(MaxPool())
	model.add(Dropout(0.1))
	model.add(Conv2D(num_kernels=64, kernel_size=3, activation=functions.relu))
	model.add(Conv2D(num_kernels=64, kernel_size=3, stride=(2, 2), activation=functions.relu))
	model.add(BatchNormalization())
	# model.add(MaxPool())
	model.add(Dropout(0.2))
	model.add(Conv2D(num_kernels=128, kernel_size=3, activation=functions.relu))
	model.add(Conv2D(num_kernels=128, kernel_size=3, stride=(2, 2), activation=functions.relu))
	model.add(BatchNormalization())
	# model.add(GlobalAveragePool())
	# model.add(MaxPool())
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(128, activation=functions.relu))
	model.add(BatchNormalization())
	model.add(Dense(no_of_actions, activation=functions.softmax))

	model.compile(optimizer=optimizers.adam,loss=functions.cross_entropy,learning_rate=0.001)
	return model
