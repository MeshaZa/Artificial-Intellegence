import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import time
def dec(func):
  def wrap(*args):
    times = time.clock()
    func(*args)
    print(time.clock() - time)
  return wrap


np.random.seed(1)
alpha = 0.005
hidden_size = 100
ans_file_name = 'ansdata.txt'
pocket_size = 100
data_amount = 201


def tan(x):
    return np.tanh(x)

def crystal(x):
   return 1/(1 + np.exp(-x))

def rest(x):
    return x*(1-x)

def softmax(x):
  temp = np.exp(x)
  return temp/np.sum(temp, axis=1, keepdims=True) 

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

inputt, output = (x_train[0:data_amount].reshape(data_amount, 28*28)/255, y_train[0:data_amount])
set_output = np.zeros((data_amount,10))
for i, j in enumerate(output):
  set_output[i][j] = 1
output = set_output
weights0_1 = 0.02*np.array(np.random.random((28*28,hidden_size)))-0.01
weights1_2 = 0.2*np.array(np.random.random((hidden_size,10)))-0.1

for amount in range(1000):
    full_err = 0
    correct_err = 0
    for i in range(int(data_amount/ pocket_size)):
        start, end = (i * pocket_size,(i+1)*pocket_size)
        layer_0 = inputt[start:end]
        layer_1 = crystal(layer_0.dot(weights0_1))
        dropout = np.random.randint(2, size = layer_1.shape) * 2
        layer_1 *= dropout
        layer_2 = softmax(layer_1.dot(weights1_2))

        delta2 = layer_2 - output[start:end]
        delta1 = delta2.dot(weights1_2.T) * rest(layer_1) * dropout

        error = np.sum(delta2) ** 2
        full_err += error

        for j in range(start, end):
          try:
            correct_err += int(np.argmax(np.array([layer_2[j]])) == np.argmax(np.array([output[j]])))
          except:
            print(j)

        weights0_1 -= layer_0.T.dot(delta1) * alpha
        weights1_2 -= layer_1.T.dot(delta2) * alpha
    if(amount%20 == 0):
      print('I: {0} err:  {1} correct: {2}%'.format( amount, round(full_err, 10), int(correct_err/data_amount * 100)))
      

