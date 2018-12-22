from sklearn.decomposition import PCA
import glove
from keras.preprocessing.text import Tokenizer
import bcolz
from matplotlib import pyplot
import os
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from gensim.parsing.preprocessing import preprocess_string
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import numpy as np
import pickle
import gc
from sklearn.decomposition import PCA
import glove
import bcolz
from matplotlib import pyplot
import os
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from gensim.parsing.preprocessing import preprocess_string
from nltk.tokenize import sent_tokenize, word_tokenize

print("Loading Glove model")
vectors = bcolz.open('glove.6B//6B.100.dat')[:]
words = pickle.load(open('glove.6B/6B.100_words.pkl', 'rb'))
word2idx = pickle.load(open('glove.6B/6B.100_idx.pkl', 'rb'))
glove = {w: vectors[word2idx[w]] for w in words}
print("model loaded")




def get_vector(word1):
	UNK = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	vector1 = []
	if "-" not in word1 and "/" not in word1 and "\'" not in word1:
		try:
			bc = glove[word1]
			vector1.append(bc)

		except Exception as inst:
			vector1 = UNK
			print("Word is", word1)
		#continue
	else:
		if "-"  in word1:
			vectors_combo = [[]]
			abcd = word1.split('-')
			try:
				vector_1 = glove[abcd[0]]
				vectors_combo.append(vector_1)
			except Exception as Inst:
				print('error occured for  ', abcd[0])
				vector1= UNK
				vectors_combo.append(vector_1)
			try:
				vector_2 = glove[abcd[1]]
				vectors_combo.append(vector_2)
			except Exception as Inst:
				print('error occured for ', abcd[1])
				vector2 = UNK
				vectors_combo.append(vector2)



			del vectors_combo[0]
			bc = [sum(x) for x in zip(*vectors_combo)]
			myInt = len(vectors_combo)
			bc[:] = [x/myInt for x in bc]
			vector1.append(bc)

		if "/"  in word1:
			vectors_combo = [[]]
			abcd = word1.split('/')
	#print(abcd)
			try:
				vector_1 = glove[abcd[0]]
				vectors_combo.append(vector_1)
			except Exception as Inst:
				print('error occured for  ', abcd[0])
				vector_1 = UNK
				vectors_combo.append(vector_1)
			try:
				vector_2 = glove[abcd[1]]
				vectors_combo.append(vector_2)
			except Exception as Inst:
				print('error occured for ', abcd[1])
				vector_2 = UNK
				vectors_combo.append(vector_2)
			del vectors_combo[0]
			bc = [sum(x) for x in zip(*vectors_combo)]
			myInt = len(vectors_combo)
			bc[:] = [x/myInt for x in bc]
			vector1.append(bc)
		if "\'" in word1:
			abcd = word1.split("\'")
			try:
				#print(" word is caught ", abcd[0])
				bc = glove[abcd[0]]
				vector1.append(bc)

			except Exception as inst:
				vector1 = UNK
				print("Word is", word1)
			
			
	return vector1



def create_embedding_matrix(tokenizer, embedding_dim):
    """
    Create embedding matrix containing word indexes and respective vectors from word vectors
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object containing word indexes
        word_vectors (dict): dict containing word and their respective vectors
        embedding_dim (int): dimention of word vector

    Returns:

    """
    nb_words = len(tokenizer.word_index) + 1
    print(' no. of words ', nb_words)
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        embedding_vector = get_vector( word)
        if embedding_vector is not None:
            embedding_matrix[i] = np.array(embedding_vector)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


def word_embed_meta_data(documents, embedding_dim):
    """
    Load tokenizer object for given vocabs list
    Args:
        documents (list): list of document

    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    print(documents[0])
    tokenizer = Tokenizer(split=' ')
    # new_doc = " ".join(documents)
    #print(new_doc[0])
    #tokenizer.fit_on_texts(" ".join(documents))
    #new_doc = ['this is a boy.', 'this was a mango.']
    tokenizer.fit_on_texts(documents)
    #print(documents)
    # print(tokenizer.word_counts)
    print(len(documents))

    
    embedding_matrix = create_embedding_matrix(tokenizer, embedding_dim)
    return tokenizer, embedding_matrix



def create_train_dev_set(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        sentences_pair (list): list of tuple of sentences pairs
        is_similar (list): list containing labels if respective sentences in sentence1 and sentence2
                           are same or not (1 if same else 0)
        max_sequence_length (int): max sequence length of sentences to apply padding
        validation_split_ratio (float): contain ratio to split training data into validation data

    Returns:
        train_data_1 (list): list of input features for training set from sentences1
        train_data_2 (list): list of input features for training set from sentences2
        labels_train (np.array): array containing similarity score for training data
        leaks_train(np.array): array of training leaks features

        val_data_1 (list): list of input features for validation set from sentences1
        val_data_2 (list): list of input features for validation set from sentences1
        labels_val (np.array): array containing similarity score for validation data
        leaks_val (np.array): array of validation leaks features
    """
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val


def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2, leaks_test


print(get_vector('king'))
print(get_vector( 'queen1'))
print(get_vector( 'king/queen1'))			

