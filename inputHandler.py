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



def create_embedding_matrix(tokenizer, embedding_dim):
    """
    This function create embedding matrix containing word indexes and respective vectors from word vectors
    Its arguments are :
    tokenizer: It is a keras tokenizer object containing word indexes
    embedding_dim (int): dimention of word vector
    It returns embedding matrix
    """
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    glove = load_model
    for word, i in word_index.items():
        embedding_vector = word_vectors[glove, word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def word_embed_meta_data(documents, embedding_dim):
    """
    This function loads tokenizer object for given vocabs list
    It takes a list of documents and the embedding dimension as argument.
    It returns tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object and embedding_matrix (dict): dict with word_index and vector mapping
    """
    tokenizer = Tokenizer(split=' ')
    tokenizer.fit_on_texts(documents)
    print(len(documents))
    embedding_matrix = create_embedding_matrix(tokenizer embedding_dim)
    return tokenizer, embedding_matrix



def create_train_dev_set(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
    """
    This function creates training and validation dataset
    Its arguments are
    1. tokenizer: keras tokenizer object
    2. sentences_pair (list): list of tuple of text pairs
    3. is_similar (list): list containing labels if respective texts in text11 and text2
       are written by same author or not (1 if same else 0)
    4. max_sequence_length (int): max sequence length of texts for padding
    5. validation_split_ratio (float): contain ratio to split training data into validation data
    It returns: 
    1. train_data_1 (list): list of input features for training set from text1
    2. train_data_2 (list): list of input features for training set from text2
    3. labels_train (np.array): array containing similarity score for training data
    4. leaks_train(np.array): array of training leaks features
    5. val_data_1 (list): list of input features for validation set from text1
    6. val_data_2 (list): list of input features for validation set from text2
    7. labels_val (np.array): array containing similarity score for validation data
    8. leaks_val (np.array): array of validation leaks features
    """
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    #In sentences1 and sentences2 the corresponding senteces are stored.
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]
    #Leaks represent the text pairs in which both text1 and text2 are same.
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
    #getting index for shuffling the data
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
    It creates training and validation dataset
    Its arguments are tokenizer: keras tokenizer object
        test_sentences_pair: list of tuple of sentences pairs
        max_sequence_length: max sequence length of sentences to apply padding
    It returns 
        test_data_1 (list): list of input features for training set from text1
        test_data_2 (list): list of input features for training set from text2
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
