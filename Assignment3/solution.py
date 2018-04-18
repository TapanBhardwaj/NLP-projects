#!/usr/bin/env python

import nltk
from nltk.corpus import wordnet

file = open("ner.txt", encoding="ISO-8859-1")
train_data = []
train_sentence = []
for line in file:
    words = line.split()
    if len(words) == 0 or words[0] == '' or words is None:
        train_data.append(train_sentence)
        train_sentence = []
    else:
        train_sentence.append(words)


# print(train_data[0])

# adding wordnet semantic label to the training data
def add_wordnet_semantic_label(train_data):
    for sentence_no in range(len(train_data)):
        for token_no in range(len(train_data[sentence_no])):
            syns = wordnet.synsets(train_data[sentence_no][token_no][0])
            if len(syns) > 0:
                train_data[sentence_no][token_no].insert(1, syns[0].lemmas()[0].name())
                syns = ''
    return train_data


def add_pos_tag(train_data):
    for sentence_no in range(len(train_data)):
        for token_no in range(len(train_data[sentence_no])):
            tag = nltk.tag.pos_tag(list(train_data[sentence_no][token_no][0]))
            train_data[sentence_no][token_no].insert(1, tag[0][1])
    return train_data


import re


def add_capital_and_digit_tag(train_data):
    for sentence_no in range(len(train_data)):
        for token_no in range(len(train_data[sentence_no])):
            r = re.match(r'[A-Z](.*)', train_data[sentence_no][token_no][0])
            d = re.match(r'([0-9]+)$', train_data[sentence_no][token_no][0])
            if r is not None:
                train_data[sentence_no][token_no].insert(1, 'CAPITAL')
            if d is not None:
                train_data[sentence_no][token_no].insert(1, 'DIGITS')

    return train_data


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def find_stop_words(train_data):
    for sentence_no in range(len(train_data)):
        for token_no in range(len(train_data[sentence_no])):
            tag = nltk.word_tokenize(train_data[sentence_no][token_no][0])
            if tag[0] in stop_words:
                train_data[sentence_no][token_no].insert(1, 'stopwords')
    return train_data


import numpy as np

# print(type(train_data))
np.random.shuffle(train_data)


# print(train_data[0])

def write_complete_training_data(train_data):
    f = open("ner_feat_new.txt", 'w')
    for sentence_no in range(len(train_data)):
        for token_no in range(len(train_data[sentence_no])):
            token_length = len(train_data[sentence_no][token_no]) - 1
            for token in train_data[sentence_no][token_no]:
                if token_length == 0:
                    f.write(str(token))
                else:
                    f.write(str(token) + " ")

                token_length -= 1
            f.write("\n")
        f.write("\n")


#to write no features, run this command:
#write_complete_training_data(train_data)

train_data = add_wordnet_semantic_label(train_data)
train_data = find_stop_words(train_data)
train_data = add_pos_tag(train_data)
train_data = add_capital_and_digit_tag(train_data)

#to write all features, run this command:
write_complete_training_data(train_data)
