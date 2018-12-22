import numpy as np
import os
import random

import sys

from model import SiameseBiLSTM
from Load_Glove import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd

#from testing import make_test_data, read_neg_data, make_new_train_data, testing1
#from testing import make_test_data, testing1
papers_list =[]
from sklearn.utils import shuffle

'''
def make_test_data():
	text1 = []
	text2 = []
	fopen1 = open('C50/C50/C50test/KarlPenhaul/37153newsML.txt', 'r')
	text1.append(fopen1.read())
	fopen2 = open('C50/C50/C50test/KarlPenhaul/49165newsML.txt', 'r')
	text2.append(fopen2.read())
	fopen11 = open('/home/mtp-2/Desktop/siamese paper/implementation/C50/C50/C50test/KarlPenhaul/471829newsML.txt' , 'r')
	text1.append(fopen11.read())
	fopen12 = open('/home/mtp-2/Desktop/siamese paper/implementation/C50/C50/C50test/KarlPenhaul/471835newsML.txt', 'r')
	text2.append(fopen12.read())
	
	text1_pair = [(x1, x2) for x1, x2 in zip(text1, text2)]
	return text1_pair


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


def data_make(Dir_name):
	k =0 
	print('In data make')
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


	auth_list =[]
	for item in papers_list :
		auth = item['paper_id']
		#print('auth',auth)
		auth1 = auth.split('/')
		auth_list.append(auth1[2])

	#print(auth_list)
	#print(len(papers_list))

	text1=[]
	text2 = []
	class1 = []
	

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
	print(len(text1))
	print(len(text2))
	print(len(class1))
	'''	
	num = 20
	num_folder = 10
	num_pair = num *(num_folder-1)
	text1_pair = [(x1, x2) for x1, x2 in zip(text1, text2)]
	print(len(text1_pair))
	#print(text_pair)
	#return text1, text2, class1, text_pair
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
						
			
	
	#print(papers_list)
	
	text_pair = [(x1, x2) for x1, x2 in zip(text11, text12)]
	#print(len(text_pair))
	#print("for no class ", text_pair)
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
		#print(list1)
		for x in range(num):
			ab = random.randint(0, len(list1)-1)
			A.append(ab)
		#print(A)
		#print('A is', A)
		for p in range(0, len(A)):
				final_text_list.append(list1[A[p]])
				class12.append(0)
		i=j+1
			
	#print('final list of no class', final_text_list)
	#print(len(final_text_list))
	#print(len(text1_pair))
	mergedlist = final_text_list + text1_pair
	#print(len(mergedlist))
	mergedclass = class1+class12
	#print(len(mergedclass))
	#print(mergedclass)
	#random.shuffle(mergedclass)
	mergedclass, mergedlist = shuffle(mergedclass, mergedlist, random_state=0)
	#print(mergedclass)
	#print(mergedlist[0][1])
	
	text_list11 = []
	text_list12 =[]
	for i in range(0, len(mergedlist)):
		text_list11.append(mergedlist[i][0])
		text_list12.append(mergedlist[i][1])
	
	#print(len(text_list11))
	#print(len(text_list12))
	return  text_list11, text_list12, mergedclass, mergedlist
	
	
k=0



sentences1, sentences2, is_similar, sentences_pair = data_make('C50/C50train')
'''
print(sentences_pair[0][0])

print(sentences_pair[0][1])


print(sentences_pair[1][0])

print(sentences_pair[1][1])

'''
print('done')
tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])

embedding_meta_data = {
	'tokenizer': tokenizer,
	'embedding_matrix': embedding_matrix
}

'''
## creating sentence pairs
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
del sentences1
del sentences2
'''


######## Training ########

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
print('go to siamese train')
siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)

best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data, model_save_directory='/home/mtp-2/Desktop/siamese paper/implementation')
print('best_model_path is ', best_model_path)

from operator import itemgetter
from keras.models import load_model

model = load_model(best_model_path)

#test_sentence_pairs = [('What can make Physics easy to learn? I am going to learn physis. I love it.','How can you make physics easy to learn? Physics is my love and i will learn it.'),('How many times a day do a clocks hands overlap? This clock is very lovely.','What does it mean that every time I look at the clock the numbers are the same? Clock looks beautiful.')]
#testing1(best_model_path)
'''
print(results)

print(preds)


test_pairs = make_test_data()
test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])
#test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])

preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
results = [(x, y, z) for (x, y), z in zip(test_pairs, preds)]
results.sort(key=itemgetter(2), reverse=True)
print results

labels = []
for i in range(0, len(preds)):
	if(preds[i]>= 0.5):
		labels.append(1)
	else:
		labels.append(0)



print(preds)
print(labels)


'''




