import pandas as pd
import numpy as np

def make_data(filename, manage_start, data_amount):
	df = pd.read_csv(filename)
	rewiews = list(map(lambda x:x.lower().strip().split(' '), df.review))[manage_start:manage_start + data_amount]
	vocab = set()

	for sent in rewiews:
		for word in sent:
			if(len(word) > 0):
				vocab.add(word)
	vocab = list(vocab)

	word2list = {}
	for i, word in enumerate(vocab):
		word2list[word] = i

	
	input_data = []
	for sent in rewiews:
		temp_data = []
		for word in sent:
			try:
				temp_data.append(word2list[word])
			except:
				pass
		input_data.append(temp_data)
	output_data = list(map(lambda x:int(x == 'Positive'), df.sentiment))[manage_start:manage_start +  data_amount]
	return [input_data, output_data, len(vocab)]