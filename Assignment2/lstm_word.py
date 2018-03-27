import string
from pickle import dump

import numpy as np
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import gutenberg
from numpy import array


# read file
def load_file(filename):
    """
    loads the filename into memory and returns the contents of the file
    :param filename:
    :return:
    """
    # open the file as read mode
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens, removing punctuations
def get_filtered_token(doc):
    """
    takes a text string and generates tokens after removing punctuation marks
    :param doc:
    :return:
    """
    # split into tokens by white space
    tokens = doc.split()

    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]

    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens


# save tokens to file
def save_file(lines, filename):
    """
    saves the content into given file_path

    :param lines:
    :param filename:
    :return:
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def perplexity(model, tokenizer):
    """
    prints the perplexity of the given model

    :param model:
    :param tokenizer:
    :return:
    """
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)
    # load
    test_filename = 'gutenberg_sequences_test.txt'
    doc = load_file(test_filename)
    lines = list(doc.split('\n'))

    all_sequences = tokenizer.texts_to_sequences(lines)
    sequences = []

    for sequence in all_sequences:
        if len(sequence) == 51:
            sequences.append(sequence)

    # separate into input and output
    sequences = np.asarray(sequences)
    print(sequences.shape)

    X_test, y_test = sequences[:, :-1], sequences[:, -1]
    y_test = to_categorical(y_test, num_classes=vocab_size)

    # evaluating model accuracy and loss
    eval_ = model.evaluate(X_test, y_test, batch_size=128, verbose=1)

    # print perplexity , eval_[0] is loss, perplexity is np.exp(loss)
    print("Perplexity of the lstm word model is {}".format(np.exp(eval_[0])))


if __name__ == '__main__':
    # taking 4 files of gutenberg corpus
    files = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt']
    raw_text = gutenberg.raw(files)

    # clean document
    tokens = get_filtered_token(raw_text)
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))

    length = 20 + 1
    sequences = list()
    for i in range(length, len(tokens)):
        # select sequence of tokens
        seq = tokens[i - length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))

    # save sequences to file
    out_filename = 'gutenberg_sequences.txt'
    save_file(sequences, out_filename)

    # loading sequence file
    in_filename = 'gutenberg_sequences.txt'
    doc = load_file(in_filename)
    lines = doc.split('\n')

    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # separate into input and output
    sequences = array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]

    # creating lstm model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X, y, batch_size=128, epochs=100)

    # save the model to file
    model.save('word_model.h5')
    # save the tokenizer
    dump(tokenizer, open('tokenizer.pkl', 'wb'))

    # print perplexity
    perplexity(model, tokenizer)
