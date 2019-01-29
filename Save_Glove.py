words = []
idx = 0
word2idx = {}
import bcolz
import numpy as np
import pickle
vectors = bcolz.carray(np.zeros(1), rootdir='glove.6B/6B.100.dat', mode='w')
'''
This file makes a pickle file after reading the Glove word file, so that at the time of finding words, we can access vectors from pickle file faster.
'''
with open('glove.6B/glove.6B.100d.txt', 'rb') as f:
    for l in f:
        line = l.decode('utf-8').split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400000, 100)), rootdir='glove.6B/6B.100.dat', mode='w')
vectors.flush()
pickle.dump(words, open('glove.6B/6B.100_words.pkl', 'wb'))
pickle.dump(word2idx, open('glove.6B/6B.100_idx.pkl', 'wb'))

print('done')

