import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def label_tokenize(train_labels, validation_labels):
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    training_label = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label = np.array(label_tokenizer.texts_to_sequences(validation_labels))
    return training_label, validation_label

def tokenize(train_tweets, validation_tweets, vocab_size, max_length):
    tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<OOV>")
    tokenizer.fit_on_texts(train_tweets)
    train_padded = padded_sequence(train_tweets, tokenizer, max_length)
    validation_padded = padded_sequence(validation_tweets, tokenizer, max_length)
    return train_padded, validation_padded

def padded_sequence(text, tokenizer, max_length):
    trunc_type='post'
    padding_type='post'
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)
    return padded