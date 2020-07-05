import emoji
import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

emoji_list = emoji.UNICODE_EMOJI.keys()
FLAGS = re.MULTILINE | re.DOTALL

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def preprocess(text):
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
    
    text = text.replace("#","<hashtag> ")
    
    no_emoji = ''
    for char in text:
        if char not in emoji_list:
            no_emoji = no_emoji + char
        else:
            no_emoji = no_emoji + '<' + emoji.demojize(char) + '> '
    text = no_emoji
    
    text = re_sub(r"@\w+","<user>")
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    text = re_sub(r"([A-Z]){2,}", allcaps)
    
    punctuations = '''!()-[]{};:'"\,./?@#$%^&*_~0123456789'''
    
    no_punct = ''
    for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
    text = no_punct
    text = text.replace("  "," ")
    text = text.replace("\n"," ")
    return text.lower()

def label_tokenize(labels, training_portion):
    divide = int(len(labels)*training_portion)
    train_labels, validation_labels = labels[:divide], labels[divide:]
    
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    training_label = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label = np.array(label_tokenizer.texts_to_sequences(validation_labels))
    return training_label, validation_label

def tokenize(text, training_portion, vocab_size, oov_tok):
    divide = int(len(text)*training_portion)
    train_tweets, validation_tweets = text[:divide], text[divide:]
    
    tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
    tokenizer.fit_on_texts(train_tweets)
    train_padded = sequence(train_tweets, tokenizer)
    validation_padded = sequence(validation_tweets, tokenizer)
    return train_padded, validation_padded

def sequence(text, tokenizer):
    max_length = 50
    trunc_type='post'
    padding_type='post'
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)
    return padded

# if __name__=="__main__":
def main():
    data = pd.read_csv("data/Twitter_Data.csv")
    labels = list(data['Annotation'])
    texts = list(data['Tweets'])
    for i in range(0, len(texts)):
        texts[i] = preprocess(texts[i])
    
    vocab_size = 18000
    oov_tok = "<OOV>"
    
    train_padded, validation_padded = tokenize(texts, .8, vocab_size, oov_tok)
    training_label, validation_label = label_tokenize(labels, .8)
    return train_padded, validation_padded, training_label, validation_label