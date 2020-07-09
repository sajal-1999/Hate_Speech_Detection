import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def get_word_index(text, vocab_size):
    tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<OOV>")
    tokenizer.fit_on_texts(text)
    return tokenizer.word_index

def embedding(text, vocab_size, embedding_dim):
    word_index = get_word_index(text, vocab_size)
    
#     embedding_dim = 200
    embeddings_index = {};
    file_name = 'glove.6B.'+ str(embedding_dim) + 'd.txt'
    with open(file_name, errors='ignore') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])#, dtype='float32')
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    
    return embeddings_matrix
