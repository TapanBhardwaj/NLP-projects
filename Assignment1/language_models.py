import math
import time
from random import shuffle

import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg

START_SYMBOL = '<s>'
STOP_SYMBOL = '</s>'
BROWN_DATA_SET = 'brown'
GUTENBERG_DATA_SET = 'gutenberg'

# n-grams counts
UNI_GRAM_COUNT = dict()
BI_GRAM_COUNT = dict()
TRI_GRAM_COUNT = dict()
FOUR_GRAM_COUNT = dict()

# n-grams probabilities
UNI_GRAM_PROB = dict()
BI_GRAM_PROB = dict()
TRI_GRAM_PROB = dict()
FOUR_GRAM_PROB = dict()

# set of unique words in corpus including stop word
TRAINING_VOCAB = set()

TOTAL_TRAIN_WORDS = 0
TOTAL_TEST_WORDS = 0


def get_n_grams(sentence, n):
    """
    Return the ngrams including starting and stop words generated from a sentence.

    :param sentence: list of words, words are string
    :param n: 2-gram , 3-gram or 4-gram
    :return: ngrams generated from a sentence
    """
    return nltk.ngrams(sentence, n, pad_left=True, left_pad_symbol=START_SYMBOL)


def katz_back_off_bigram(word, word_hist, gamma=0.5):
    # set_a = set()
    # for bi_gram in BI_GRAM_COUNT:
    #     if bi_gram[0] == word_hist:
    #         set_a.add(bi_gram[1])
    #
    # set_b = TRAINING_VOCAB.difference(set_a)
    #
    # if word in set_a:
    return None


def perplexity_uni_gram(test_data, total_test_words):
    """ returns perplexity for uni-gram, using back-off"""
    total_prob_sum = 0

    for i in range(len(test_data)):

        for uni_gram in test_data[i]:
            if uni_gram in UNI_GRAM_PROB:
                total_prob_sum += UNI_GRAM_PROB[uni_gram]
            else:
                total_prob_sum -= math.log(TOTAL_TRAIN_WORDS, 2)

    perplexity = math.pow(2, -1 * (total_prob_sum / total_test_words))

    print("Total perplexity of uni_gram model is : {}".format(perplexity))


def perplexity_bi_gram(test_data, total_test_words):
    """ returns perplexity for bi-gram, using back-off"""
    total_prob_sum = 0

    for i in range(len(test_data)):

        bi_grams = get_n_grams(test_data[i], 2)

        for bi_gram in bi_grams:
            if bi_gram in BI_GRAM_PROB:
                total_prob_sum += BI_GRAM_PROB[bi_gram]
            elif bi_gram[1] in UNI_GRAM_PROB:
                total_prob_sum += UNI_GRAM_PROB[bi_gram[1]]
            else:
                total_prob_sum += math.log(1 / TOTAL_TRAIN_WORDS, 2)

    perplexity = math.pow(2, -1 * (total_prob_sum / total_test_words))

    print("Total perplexity of bi_gram model is : {}".format(perplexity))


def perplexity_trigram(test_data, total_test_words):
    """ returns perplexity for tri-gram, using back-off"""
    total_prob_sum = 0

    for i in range(len(test_data)):

        trigrams = get_n_grams(test_data[i], 3)

        for trigram in trigrams:
            if trigram in TRI_GRAM_PROB:
                total_prob_sum += TRI_GRAM_PROB[trigram]
            elif trigram[1:] in BI_GRAM_PROB:
                total_prob_sum += BI_GRAM_PROB[trigram[1:]]
            elif trigram[2] in UNI_GRAM_PROB:
                total_prob_sum += UNI_GRAM_PROB[trigram[2]]
            else:
                total_prob_sum -= math.log(TOTAL_TRAIN_WORDS, 2)

    perplexity = math.pow(2, -1 * (total_prob_sum / total_test_words))

    print("Total perplexity of trigram model is : {}".format(perplexity))


def perplexity_four_gram(test_data, total_test_words):
    """ returns perplexity for four-gram, using back-off """
    total_prob_sum = 0

    for i in range(len(test_data)):

        four_grams = get_n_grams(test_data[i], 4)

        for four_gram in four_grams:
            if four_gram in FOUR_GRAM_PROB:
                total_prob_sum += FOUR_GRAM_PROB[four_gram]
            elif four_gram[1:] in TRI_GRAM_PROB:
                total_prob_sum += TRI_GRAM_PROB[four_gram[1:]]
            elif four_gram[2:] in BI_GRAM_PROB:
                total_prob_sum += BI_GRAM_PROB[four_gram[2:]]
            elif four_gram[3] in UNI_GRAM_PROB:
                total_prob_sum += UNI_GRAM_PROB[four_gram[3]]
            else:
                total_prob_sum -= math.log(TOTAL_TRAIN_WORDS, 2)

    perplexity = math.pow(2, -1 * (total_prob_sum / total_test_words))

    print("Total perplexity of four_gram model is : {}".format(perplexity))


def calculate_n_grams_probability():
    """
    returns a dict having probability for each n-gram
    :return: return a dict having probability for each n-gram
    """
    # Count the total number of words in corpus, without START_SYMBOL but include STOP_SYMBOL
    total__words = 0

    for uni_gram in UNI_GRAM_COUNT:
        total__words += UNI_GRAM_COUNT[uni_gram]

    log_n = math.log(total__words, 2)
    for uni_gram in UNI_GRAM_COUNT:
        UNI_GRAM_PROB[uni_gram] = math.log(UNI_GRAM_COUNT[uni_gram], 2) - log_n

    for bi_gram in BI_GRAM_COUNT:
        BI_GRAM_PROB[bi_gram] = math.log(BI_GRAM_COUNT[bi_gram], 2) - math.log(UNI_GRAM_COUNT[bi_gram[0]], 2)

    for trigram in TRI_GRAM_COUNT:
        TRI_GRAM_PROB[trigram] = math.log(TRI_GRAM_COUNT[trigram], 2) - math.log(BI_GRAM_COUNT[trigram[:2]], 2)

    for four_gram in FOUR_GRAM_COUNT:
        FOUR_GRAM_PROB[four_gram] = math.log(FOUR_GRAM_COUNT[four_gram], 2) - math.log(
            TRI_GRAM_COUNT[four_gram[:3]], 2)

    del UNI_GRAM_COUNT[START_SYMBOL]
    del BI_GRAM_COUNT[(START_SYMBOL, START_SYMBOL)]
    del TRI_GRAM_COUNT[(START_SYMBOL, START_SYMBOL, START_SYMBOL)]


def calculate_n_grams_count(training_data):
    """
    setting the global counts of uni_gram, bi_gram, trigram, four_gram

    :param training_data: list of sentences using to train model
    """
    # stores counts of different n-grams
    global UNI_GRAM_COUNT
    global BI_GRAM_COUNT
    global TRI_GRAM_COUNT
    global FOUR_GRAM_COUNT

    total_train_sentences = len(training_data)

    for sentence in training_data:

        for token in sentence:
            if token in UNI_GRAM_COUNT:
                UNI_GRAM_COUNT[token] += 1
            else:
                UNI_GRAM_COUNT[token] = 1

        UNI_GRAM_COUNT[START_SYMBOL] = total_train_sentences

    # calculating counts for different n grams
    for sentence in training_data:

        for bi_gram in get_n_grams(sentence, 2):
            if bi_gram in BI_GRAM_COUNT:
                BI_GRAM_COUNT[bi_gram] += 1
            else:
                BI_GRAM_COUNT[bi_gram] = 1

        BI_GRAM_COUNT[(START_SYMBOL, START_SYMBOL)] = total_train_sentences

        for trigram in get_n_grams(sentence, 3):
            if trigram in TRI_GRAM_COUNT:
                TRI_GRAM_COUNT[trigram] += 1
            else:
                TRI_GRAM_COUNT[trigram] = 1

        TRI_GRAM_COUNT[(START_SYMBOL, START_SYMBOL, START_SYMBOL)] = total_train_sentences

        for four_gram in get_n_grams(sentence, 4):
            if four_gram in FOUR_GRAM_COUNT:
                FOUR_GRAM_COUNT[four_gram] += 1
            else:
                FOUR_GRAM_COUNT[four_gram] = 1


def main(data_set):
    """
    main function to implement language models. bi-gram, tri-gram , four-gram

    :param data_set: either BROWN_DATA_SET or GUTENBERG_DATA_SET
    :return:
    """
    global TRAINING_VOCAB
    global TOTAL_TRAIN_WORDS
    global TOTAL_TEST_WORDS
    sentences = list()

    if data_set == BROWN_DATA_SET:
        sentences = list(brown.sents(brown.fileids()))
        TRAINING_VOCAB = set(w.lower() for w in brown.words(brown.fileids()))
    elif data_set == GUTENBERG_DATA_SET:
        sentences = list(gutenberg.sents(gutenberg.fileids()))
        TRAINING_VOCAB = set(w.lower() for w in gutenberg.words(gutenberg.fileids()))

    TRAINING_VOCAB.add(STOP_SYMBOL)

    # converting all sentences to lower case
    for i in range(len(sentences)):
        sentences[i] = list(map(lambda x: x.lower(), sentences[i]))

    # adding STOP_SYMBOL at the end of every sentence, and removing '.'
    for sent in sentences:
        if sent[-1] == '.':
            sent[-1] = STOP_SYMBOL
        else:
            sent.append(STOP_SYMBOL)

    # dividing data into training and test
    shuffle(sentences)
    sentences_count = len(sentences)
    train_sentences = sentences[:int(sentences_count * .9)]

    test_sentences = sentences[int(sentences_count * .9):]

    for i in range(len(train_sentences)):
        TOTAL_TRAIN_WORDS += len(train_sentences[i])

    for i in range(len(test_sentences)):
        TOTAL_TEST_WORDS += len(test_sentences[i])

    # calculate count of different n-grams
    calculate_n_grams_count(train_sentences)

    # calculate probabilities of n-grams
    calculate_n_grams_probability()

    print("Perplexity on TRAIN DATA : ")
    perplexity_uni_gram(train_sentences, TOTAL_TRAIN_WORDS)
    perplexity_bi_gram(train_sentences, TOTAL_TRAIN_WORDS)
    perplexity_trigram(train_sentences, TOTAL_TRAIN_WORDS)
    perplexity_four_gram(train_sentences, TOTAL_TRAIN_WORDS)

    print("Perplexity on TEST DATA : ")
    perplexity_uni_gram(test_sentences, TOTAL_TEST_WORDS)
    perplexity_bi_gram(test_sentences, TOTAL_TEST_WORDS)
    perplexity_trigram(test_sentences, TOTAL_TEST_WORDS)
    perplexity_four_gram(test_sentences, TOTAL_TEST_WORDS)


if __name__ == '__main__':
    # start timer
    time.clock()
    # use arg : BROWN_DATA_SET for brown corpus and GUTENBERG_DATA_SET for gutenberg corpus
    main(GUTENBERG_DATA_SET)
    print("Total time taken: " + str(time.clock()) + ' sec')
