import pandas as pd
import numpy as np
import re
from collections import Counter
import math

word2list = {}
list2word = {}

def make_data(filename, aver, data_amount):
	index = data_amount//2
	df = pd.read_csv(filename)
	rewiews = list(map(lambda x:re.findall(r"[\w']+", x.lower().strip()), df.review))[aver - data_amount:aver+data_amount]
	vocab = set()
	for sent in rewiews:
		for word in sent:
			if(len(word) > 0):
				vocab.add(word)
	vocab = list(vocab)


	for i, word in enumerate(vocab):
		word2list[word] = i
		list2word[i] = word

	
	input_data = []
	for sent in rewiews:
		temp_data = []
		for word in sent:
			try:
				temp_data.append(word2list[word])
			except:
				pass
		input_data.append(temp_data)
	output_data = list(map(lambda x:int(x == 'Positive'), df.sentiment))[aver - data_amount:aver+data_amount]
	return [input_data[index:], output_data[:index], input_data[:index], output_data[index:], len(vocab)]

def find_same_word(target, weights):
	tindex = word2list[target]
	score = Counter()
	for word, index in word2list.items():
		dif = (weights[index] - weights[tindex])**2
		score[word] = -math.sqrt(sum(dif))
	print(score.most_common(50))



