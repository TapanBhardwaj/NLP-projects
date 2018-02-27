import random
import string
from collections import defaultdict

import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg

START_SYMBOL = '<s>'
STOP_SYMBOL = '</s>'
UNK_SYMBOL = 'u_n_k'
BROWN_DATA_SET = 'brown'
GUTENBERG_DATA_SET = 'gutenberg'


def remove_punctuations(list_sentences):
    """

    :param list_sentences:
    :return:
    """
    sent_no_punc_marks = []
    punc_marks = string.punctuation + "''" + '``' + '--'

    for i in range(len(list_sentences)):
        processed_sent = []

        for k in range(len(list_sentences[i])):
            if list_sentences[i][k] not in punc_marks:
                processed_sent.append(list_sentences[i][k])

        sent_no_punc_marks.append(processed_sent)

    return sent_no_punc_marks


def get_n_grams(sentence, n):
    """
    Return the ngrams including starting and stop words generated from a sentence.

    :param sentence: list of words, words are string
    :param n: 2-gram , 3-gram or 4-gram
    :return: ngrams generated from a sentence
    """
    return nltk.ngrams(sentence, n, pad_left=True, left_pad_symbol=START_SYMBOL)


def add_stop_symbol(sentences):
    """ adding STOP_SYMBOL at the end of every sentence, and removing '.' """
    for sent in sentences:
        if sent[-1] == '.':
            sent[-1] = STOP_SYMBOL
        else:
            sent.append(STOP_SYMBOL)


def get_model_dict(training_corpus, n):
    """
    returns a tuple of dict objects (unigrams, bigrams, trigrams) that map from n-grams to counts
    :param training_corpus:
    :param n:
    :return:
    """
    model_dict = defaultdict(lambda: defaultdict(lambda: 0))

    for sentence in training_corpus:
        for n_gram in get_n_grams(sentence, n):
            model_dict[n_gram[:-1]][n_gram[-1]] += 1
    # Let's transform the counts to probabilities
    for key in model_dict:
        total_count = float(sum(model_dict[key].values()))
        for next_gram in model_dict[key]:
            model_dict[key][next_gram] /= total_count

    return model_dict


def generate_random_sentence(corpus, n):
    """
    generates random sentences using n-gram model

    :param corpus:
    :param n:
    :return:
    """
    model_dict = get_model_dict(corpus, n)

    while True:
        text = [START_SYMBOL] * (n - 1)
        sentence_finished = False

        while not sentence_finished:
            r = random.random()
            accumulator = .0

            for word in model_dict[tuple(text[-(n - 1):])].keys():
                accumulator += model_dict[tuple(text[-(n - 1):])][word]

                if accumulator >= r:
                    text.append(word)
                    break

            if text[-1] == STOP_SYMBOL:
                sentence_finished = True

        if len(text) >= 9 + n:
            sentence = ' '.join([t for t in text[:9 + n] if t not in [STOP_SYMBOL, START_SYMBOL]])
            print(sentence)
            break


def main(data_set, n=4):
    """
    predicts a sentence of 10 words using n-gram model, according to n
    :param data_set:
    :param n:
    :return:
    """
    if data_set == BROWN_DATA_SET:
        sentences = list(brown.sents(brown.fileids()))
    elif data_set == GUTENBERG_DATA_SET:
        sentences = list(gutenberg.sents(gutenberg.fileids()))

    # adding stop symbols
    add_stop_symbol(sentences)

    # remove punctuations
    sentences = remove_punctuations(sentences)

    # helper function to generate a sentence of 10 words
    generate_random_sentence(sentences, n)


if __name__ == '__main__':
    # use arg : lm.BROWN_DATA_SET for brown corpus and lm.GUTENBERG_DATA_SET for gutenberg corpus
    main(BROWN_DATA_SET, 4)  # using 4-gram
