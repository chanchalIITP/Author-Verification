from operator import itemgetter
from keras.models import load_model
from Load_Glove import word_embed_meta_data, create_test_data
import numpy as np
import os
import random
import sys
from model import SiameseBiLSTM
from Load_Glove import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd
from sklearn.utils import shuffle

papers_list =[]
best_model_path = 'lstm_50_200_0.17_0.25.h5'

def read_neg_data():
	'''
	This function read negative data for training the neural network so that it can update itself for the new authors.
	From the command line we have read the documents written by the author, so we the YES class author.
	now in this function we are trying to read some documents not written by the author.
	This function returns the text list containing all text documents form the NO class.
	'''
	text =[]
	f1 = open('29059newsML.txt', 'r')
	text.append(f1.read())
	f2 = open('30089newsML.txt' , 'r')
	text.append(f2.read())
	f3 = open('47025newsML.txt', 'r')
	text.append(f3.read())
	return text

	
	

def make_new_train_data():
	'''
	In this function we are creating new training data for the questioned author.
	First we are reading the neg data.
	Then from the command line we are reading positive data, and the unknown data.
	'''
	# neg_text reads the negative documents from the read_neg_data function.
	neg_text = read_neg_data()
	#reading documents name from arguments
	arguments = sys.argv[1:]
	# finding number of documents
	num_documents = len(arguments) 
	#reading train documents from command line
	text = []
	for i in range(1, num_documents):
		f = open(sys.argv[i],"r")
		contents = f.read()
		text.append(contents)
		f.close()
		
	#for storing yes class	
	# text11 for reading the first text of the pair
	#text12 for reading the second text of pair
	#class1 is for storing the class name
	text11 = []
	text12 = []
	class1=[]
	for i in range(0, len(text)):
		for j in range(0, len(text)):
			text11.append(text[i])
			text12.append(text[j])
			class1.append(1)
			
	#for negative class		
	# text21 for reading the first text of the pair
	#text22 for reading the second text of pair
	#class2 is for storing the class name
	
	text21 = []
	text22 = []
	class2 = []
	for i in range(0, len(text)):
		for j in range(0, len(neg_text)):
			text21.append(text1[i])
			text22.append(neg_text[j])
			class2.append(0)
	#appending the both class's corresponding pair 	
	texts = text11+text21
	texts1 = text12+text22
	classes = class1+class2
	#making text pair
	text_pair = [(x1, x2) for x1, x2 in zip(texts, texts1)]	
	return texts, texts1, classes, text_pair



def make_test_data():
	'''
	This function makes test data.
	It reads the known documents from the comaandline in text1,
	It reads reads the unknown document in text2.
	and then pairs it out as:
	known1 , unknown
	known2, unknown
	known3, unknown
	finally it returns the texts and text pairs.
	'''
	arguments = sys.argv[1:]
	num_documents = len(arguments) 
	text1 = []
	#reading known document
	for i in range(1, num_documents):
		f = open(sys.argv[i],"r")
		contents = f.read()
		text1.append(contents)
		f.close()
	#reading unknown document
	text2 = []
	f = open(sys.argv[num_documents])
	contents = f.read()
	f.close()
	for i in range(0, num_documents-1):
		text2.append(contents)
	#making text pair
	text_pair = [(x1, x2) for x1, x2 in zip(text1, text2)]	
	return text_pair






def testing1(best_model_path ):
	#making the training data
	text1, text2, class1, train_pair = make_new_train_data()
	#making test data
	test_pair = make_test_data()
	#making tokenizer and emedding matrix
	tokenizer, embedding_matrix = word_embed_meta_data(text1 + text2,  siamese_config['EMBEDDING_DIM'])
	
	embedding_meta_data = {
		'tokenizer': tokenizer,
		'embedding_matrix': embedding_matrix
	}


	class Configuration(object):
	    """Dump stuff here"""

	CONFIG = Configuration()
	# setting configuration for the model
	CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
	CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
	CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
	CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
	CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
	CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
	CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
	CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']
	# making siamese network
	siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)
	#updating the pretrained model and saving it into the model.
	best_model_path = siamese.update_model(best_model_path, train_pair, class1, embedding_meta_data)
	# loading the best updated model
	model = load_model(best_model_path)
	# creatng text data as per requirement
	test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_pair,  siamese_config['MAX_SEQUENCE_LENGTH'])
	# storing results of test data in the preds varibale
	preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
	'''
	storing the results in the following form:
	known1, unknown, result2
	known2, unknown, result2
	known3, unknown , result3
	'''
	results = [(x, y, z) for (x, y), z in zip(test_pair, preds)]
	results.sort(key=itemgetter(2), reverse=True)
	return results, preds



def making_Yes_NO(preds):
	labels = []
	for i in range(0, len(preds)):
		if(preds[i]>= 0.5):
			labels.append(1)
		else:
			labels.append(0)
	return labels


def comp_labels(labels):
	for i in range(0, len(labels)):
		if(labels[i] == 1):
			print(' according to document ', i, 'the result is yes')
		else:
			print(' according to document ', i, 'the result is no')


def taking_majority(labels):
	count_0=0
	count_1=0
	for label in labels:
		if label== 1:
			count_1= count_1+1
		else:
			count_0= count_0+1
	
	print('According to ', count_1, ' document the unknown document belongs to the same author.')
	print('According to ', count_0, ' document the unknown docuemnt does not belong to the same author.')
	
	if(count_1 >= 1):
		print('the final result say yes')
	else:
		print('The final result say no.')

		

def main():	
	results, preds = testing1(best_model_path)
	labels = making_Yes_NO(preds)
	comp_labels(labels)
	taking_majority(labels)

if __name__ == "__main__":
	main()
