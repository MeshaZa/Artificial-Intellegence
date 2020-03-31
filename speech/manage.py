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

	template = [0 for i in range(len(vocab))]
	input_data = list()
	for sent in rewiews:
		temp_data = template.copy()
		for word in sent:
			try:
				temp_data[word2list[word]] = 1
			except:
				pass
		input_data.append(temp_data)

	output_data = list(map(lambda x:int(x == 'Positive'), df.sentiment))[manage_start:manage_start +  data_amount]
	return [np.array(input_data), np.array(output_data)]