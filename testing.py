from operator import itemgetter
from keras.models import load_model
from Load_Glove import word_embed_meta_data, create_test_data
import numpy as np
import os
import random
import numpy as np
import os
import random

import sys

from model import SiameseBiLSTM
from Load_Glove import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd


papers_list =[]
from sklearn.utils import shuffle
import sys

from model import SiameseBiLSTM
from Load_Glove import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd


papers_list =[]
from sklearn.utils import shuffle



best_model_path = 'lstm_50_200_0.17_0.25.h5'
model = load_model(best_model_path)

def read_neg_data():
	text22 =[]
	f1 = open('29059newsML.txt', 'r')
	text22.append(f1.read())
	f2 = open('30089newsML.txt' , 'r')
	text22.append(f2.read())
	f3 = open('47025newsML.txt', 'r')
	text22.append(f3.read())
	#f4 = open('C50/C50/C50test/KeithWeir/47027newsML.txt', 'r')
	#text22.append(f4.read())
	#f5 = open('C50/C50/C50test/KeithWeir/48329newsML.txt', 'r')
	#text22.append(f5.read())
	#f6 = open('C50/C50/C50test/KeithWeir/48360newsML.txt', 'r')
	#text22.append(f6.read())
	print('text22 is', text22)
	return text22

	
	

def make_new_train_data():
	text221 = read_neg_data()
	arguments = sys.argv[1:]
	num_documents = len(arguments) 
	text1 = []
	#print(num_documents)
	for i in range(1, num_documents):
		f = open(sys.argv[i],"r")
		contents = f.read()
		text1.append(contents)
		f.close()
	text11 = []
	text12 = []
	class1=[]
	for i in range(0, len(text1)):
		for j in range(0, len(text1)):
			text11.append(text1[i])
			text12.append(text1[j])
			class1.append(1)
	#print(text1)
	text21 = []
	text22 = []
	class2 = []
	f = open(sys.argv[num_documents])
	contents = f.read()
	f.close()
	for i in range(0, len(text1)):
		for j in range(0, len(text221)):
			text21.append(text1[i])
			text22.append(text221[j])
			class2.append(0)
		
	texts = text11+text21
	texts1 = text12+text22
	classes = class1+class2
	#print(text2)
	text_pair = [(x1, x2) for x1, x2 in zip(texts, texts1)]
	print('text_pairs', text_pair)
	print('classes', classes)
	#print(text_pair)	
	return texts, texts1, classes, text_pair



def make_test_data():

	arguments = sys.argv[1:]
	num_documents = len(arguments) 
	text1 = []
	#print(num_documents)
	for i in range(1, num_documents):
		f = open(sys.argv[i],"r")
		contents = f.read()
		text1.append(contents)
		f.close()
	#print(text1)
	text2 = []
	f = open(sys.argv[num_documents])
	contents = f.read()
	f.close()
	for i in range(0, num_documents-1):
		text2.append(contents)
	#print(text2)
	text_pair = [(x1, x2) for x1, x2 in zip(text1, text2)]
	#print(text_pair)	
	return text_pair





'''
test_sentence_pairs = [('What can make Physics easy to learn?','How can you make physics easy to learn?'),('How many times a day do a clocks hands overlap?','What does it mean that every time I look at the clock the numbers are the same?')]
sentences1 = []
sentences2 = []

for i in range(0, len(test_sentence_pairs)):
	sentences1 = test_sentence_pairs[i][0]
	sentences2 = test_sentence_pairs[i][1]
'''

def testing1(best_model_path ):
	sentences1, sentences2, class1, train_pair = make_new_train_data()

	test_pair = make_test_data()

	print('sentences1' , sentences1)
	tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])

	embedding_meta_data = {
		'tokenizer': tokenizer,
		'embedding_matrix': embedding_matrix
	}


	class Configuration(object):
	    """Dump stuff here"""

	CONFIG = Configuration()

	CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
	CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
	CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
	CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
	CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
	CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
	CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
	CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']
	print('go to siamese')
	siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)


	best_model_path = siamese.update_model(best_model_path, train_pair, class1, embedding_meta_data)

	print(best_model_path)
	from operator import itemgetter
	from keras.models import load_model

	model = load_model(best_model_path)



	test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_pair,  siamese_config['MAX_SEQUENCE_LENGTH'])

	preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
	results = [(x, y, z) for (x, y), z in zip(test_pair, preds)]
	results.sort(key=itemgetter(2), reverse=True)
	print(results)

	print(preds)
	return results, preds


results, preds = testing1(best_model_path)

print(results)

print(preds)


labels = []
for i in range(0, len(preds)):
	if(preds[i]>= 0.5):
		labels.append(1)
	else:
		labels.append(0)



print(preds)
print(labels)



def taking_majority(labels):
	count_0=0
	count_1=0
	for label in labels:
		if label== 1:
			count_1= count_1+1
		else:
			count_0= count_0+1
	
	print('Accordint to ', count_1, ' documents the unknown document belongs to the same author.')
	print('Accordint to ', count_0, ' documents the unknown docuemnt does not belong to the same author.')
	
	if(count_1 >= 1):
		print('the final result say yes')
	else:
		print('The final result say no.')


for i in range(0, len(labels)):
	if(labels[i] == 1):
		print(' according to document ', i, 'the result is yes')
	else:
		print(' according to document ', i, 'the result is no')

taking_majority(labels)
