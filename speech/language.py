import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from manage import * 

data_amount = 1000
aver = 5082
input_data, output_data = make_data('~/Desktop/MFNN/Speech/Review.csv',aver - data_amount//2, data_amount)
tinput_data, toutput_data = make_data('~/Desktop/MFNN/Speech/Review.csv',aver - data_amount//2, data_amount)
layer0_size = len(input_data[0])
hidden_size = 100
layer2_size = 1
iterations = 100
alpha = 0.01

def sigmoid(x):
	return 1/(1+np.exp(-x))
def deriv(x):
	return output*(1-output)

weights0_1 = 2*np.array(np.random.random((layer0_size,hidden_size)))-1
weights1_2 = 2*np.array(np.random.random((hidden_size,layer2_size)))-1



for amount in range(iterations):
	full_err = 0
	count = 0
	for j in range(data_amount):
		layer_0 = input_data[j:j+1]
		layer_1 = sigmoid(layer_0.dot(weights0_1))
		layer_2 = layer_1.dot(weights1_2)
		delta = layer_2 - output_data[j:j+1]
		count += int(delta < 0.5)
		hid_delta = delta.dot(weights1_2.T) 
		weights0_1 -= layer_0.T.dot(hid_delta) * alpha
		weights1_2 -= layer_1.T.dot(delta) * alpha
		err = delta ** 2
		full_err+=err
	if(amount%20==0):
		print('I:', amount,"| Err:", full_err, '| per:', count*100 / data_amount )

print(weights0_1[0], weights0_1[1])