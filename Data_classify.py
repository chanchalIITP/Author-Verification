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

def data_make(Dir_name):
	'''
	This function reads data from the folder name, which is given in the argument 'Dir_name'.
	First it makes a dictionary in which all docuemnts corresponding to an author (key is the author) is listed.
	Then data pairing is done, like  " text1 , text2 , Yes'
	It means text1 and text2 belong to the same author
	For No class, the documents from other authors are taken. 
	For balancing the number of samples in both the classes, the no. of documents is kept same.
	Finally It returns the text1, text2, class and merged text list. 
	'''
	k=0
	'''
	For reading data from the directory name
	'''
	for root, dirs,files in os.walk(Dir_name):
		if(k==0):
			k = k+1
			continue

		text_list = []
		papers = {}
		path = os.path.join(root)
		papers['paper_id'] = path
		for name in files:
			path1 = os.path.join(root, name)
			fopen = open(path1,'r')
			text = fopen.read()
			text_list.append(text)
		papers['text'] = text_list
		papers_list.append(papers)
	
	
	#papers_list contains a list of dictionary in which key is the author name and value is their text documents.
	
	auth_list =[]
	for item in papers_list :
		auth = item['paper_id']
		#print('auth',auth)
		auth1 = auth.split('/')
		auth_list.append(auth1[2])

	#auth_list contains the name of authors
	
	text1=[]
	text2 = []
	class1 = []
	'''
	text1 and text2 ae two list for keeping the documents for the Yes class. text1 and text2 stores the documents which should be paired according to their index.
	Like text1 [0] is paired with text2[0]
	class1 is to store the class name.
	'''

	for i in range(0, len(papers_list)):
		item = papers_list[i]
		auth1 = item['paper_id']
		text11 = item['text']
		for j in range(0, len(text11)):
			for k in range(0, len(text11)):
				text1.append(text11[j])
				text2.append(text11[k])
				class1.append(1)
	'''
	num is the no. of documetns per author
	num_folder is the number of different authors
	num_air = total no. of 'NO' pair which will be generated for 1 author
	'''
	num = 20
	num_folder = 10
	num_pair = num *(num_folder-1)
	# text1_pair is a list of pairs of text, crerated from text1 and text2. It is the text pair of YES class.
	
	text1_pair = [(x1, x2) for x1, x2 in zip(text1, text2)]
	
	'''
	text11 and text12 ae two list for keeping the documents for the NO class. text11 and text12 stores the documents which should be paired according to their index.
	Like text11 [0] is paired with text12[0]
	class12 is to store the class name.
	'''
	
	text11 = []
	text12 = []
	class12 = []
	for i in range(0, len(papers_list)):
		item= papers_list[i]
		text1_list = item['text']
		for j in range(0, len(text1_list)):
			for k in range(0, len(papers_list)):
				if(i == k):
					continue
				else:
					item11 = papers_list[k]
					text2_list  = item11['text']
					for l in range(0, len(text2_list)):
						text11.append(text1_list[j])
						text12.append(text2_list[l])
						
	
	text_pair = [(x1, x2) for x1, x2 in zip(text11, text12)]
	# text_pair is a list of pairs of text, crerated from text1 and text2. It is the text pair of YES class.
	final_text_list = []
	i = 0
	while(i != len(text_pair)):
		list1=[]
		#print(' i is', i)
		for j in range(i, i+num_pair):
				#print(j)
				
				if(j <len(text_pair)):
					list1.append(text_pair[j])
		
		A=[]
		# it is done to balance the no. of samples in NO class
		# the negative pair for each author is chosen  randomly equal to the positive pair.
		for x in range(num):
			ab = random.randint(0, len(list1)-1)
			A.append(ab)
		#A is the list containg the index of chosen pairs fromn the negative pair list.
		for p in range(0, len(A)):
				final_text_list.append(list1[A[p]])
				class12.append(0)
		i=j+1
		
	#Now final_text_list contains equal negative pairs.
	# postive and negative pairs are merged and shuffeled.
	# mergedclss is the class.
	# mergedlist is the list for data.
	mergedlist = final_text_list + text1_pair
	mergedclass = class1+class12
	mergedclass, mergedlist = shuffle(mergedclass, mergedlist, random_state=0)
	# text_list11 and text_list12 are the list of corresponding documents of the pair.
	# text1, text2, class : It will be the final form of data to be used for model generation.
	# text_list11 contains all the documents in text1, and text_list12 contains all the documents in text2.
	
	text_list11 = []
	text_list12 =[]
	for i in range(0, len(mergedlist)):
		text_list11.append(mergedlist[i][0])
		text_list12.append(mergedlist[i][1])
	return  text_list11, text_list12, mergedclass, mergedlist
	
	
#calling data_make function

sentences1, sentences2, is_similar, sentences_pair = data_make('C50/C50train')
# sentences1 keeps the list of text1
# sentences2 keeps the list of text2
#is_similar keeps the list of classes
#sentences_pair keeps the list of text_pair

tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])

embedding_meta_data = {
	'tokenizer': tokenizer,
	'embedding_matrix': embedding_matrix
}


######## Training ########

class Configuration(object):
    """Dump stuff here"""

CONFIG = Configuration()
#setting configuration of the siamese network

CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)
# siames is the siamese architecture
# best model path return the trained model path
best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data, model_save_directory='/home/mtp-2/Desktop/siamese paper/implementation')
print('fitted model path is ', best_model_path)




