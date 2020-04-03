import pandas as pd
import sys,random,math
from collections import Counter
import numpy as np


np.random.seed(1)
random.seed(1)

vocab = set()
word2index = {}
index2word = {}

df = list(map(lambda x:x.lower().replace(".", "").replace(",", "").replace('"', "").split(), pd.read_csv('sent_data.csv')['sentence']))

for sent in df:
    for word in sent:
        vocab.add(word)
        
vocab = list(vocab)

for i, word in enumerate(vocab):
    word2index[word] = i
    index2word[i] = word
input_dataset = []
concatenated = []


for sent in df:
    for word in range(len(sent)):
        if(word > 1 and word < len(sent)-2):
            input_dataset.append([ word2index[sent[word-2]]  ,word2index[sent[word-1]]  ,word2index[sent[word]]  ,word2index[sent[word+1]]  ,word2index[sent[word+2]]])
        concatenated.append(word2index[sent[word]])
   
    
concatenated = np.array(concatenated)
input_dataset = np.array(input_dataset)

web_improvement, alpha, sent_size, hidden_layer = (5, 0.05, 5, 50)
its = 3
web_1 = [i for i in range(sent_size) if(i!=sent_size//2)]
ans = np.zeros(web_improvement+1)
ans[0] = 1

weights_0_1 = (np.random.rand(len(vocab),hidden_layer) - 0.5) * 0.2   
weights_1_2 = np.random.rand(len(vocab), hidden_layer)*0

def similar(target='beautiful'):
  target_index = word2index[target]

  scores = Counter()
  for word,index in word2index.items():
    raw_difference = weights_0_1[index] - (weights_0_1[target_index])
    squared_difference = raw_difference * raw_difference
    scores[word] = -math.sqrt(sum(squared_difference))
  return scores.most_common(10)
def sigmoid(x):
    return 1/(1 + np.exp(-x))


for it in range(its):
    for rev_i,review in enumerate(input_dataset):
        web_2 = [review[sent_size//2]]+list(concatenated[(np.random.rand(web_improvement)*len(concatenated)).astype('int').tolist()])
        layer_1 = np.mean(weights_0_1[review[web_1]], axis = 0)
        layer_2 = sigmoid(layer_1.dot(weights_1_2[web_2].T))
        delta_2 = layer_2 - ans
        delta_1 = delta_2.dot(weights_1_2[web_2])
        weights_0_1[review[web_1]] -= delta_1 * alpha
        weights_1_2[web_2] -= np.outer(delta_2, layer_1) * alpha  


        if(rev_i%250==0):
            l = len(input_dataset)
            f = ((rev_i + l*it)*100// (l * its))
            sys.stdout.write('\r Completed: '+ "█"*(f//10) + "."*(10-f//10)+' '+str(f) + '% ')
while True:
    s = input()
    if(s == 'quit()'):
        break
    try:
        sys.stdout.write("\r" + str(similar(s)))
    except:
        sys.stdout.write("\rSorry, i don't know word"+str(s))



