import emoji
import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

emoji_list = emoji.UNICODE_EMOJI.keys()
FLAGS = re.MULTILINE | re.DOTALL

def preprocess(text):
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
    
    def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"

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

def preprocessing(filename):#, max_length, vocab_size):
    data = pd.read_csv(filename)
    labels = list(data['Annotation'])
    texts = list(data['Tweets'])
    for i in range(0, len(texts)):
        texts[i] = preprocess(texts[i])
    return texts, labels
#     train_padded, validation_padded = tokenize(texts, .8, vocab_size, max_length)
#     training_label, validation_label = label_tokenize(labels, .8)
#     return train_padded, validation_padded, training_label, validation_label