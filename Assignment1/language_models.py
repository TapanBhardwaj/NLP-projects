import math
from random import shuffle

import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg

START_SYMBOL = '<s>'
STOP_SYMBOL = '</s>'
BROWN_DATA_SET = 'brown'
GUTENBERG_DATA_SET = 'gutenberg'
UNI_GRAM_COUNT = dict()
BI_GRAM_COUNT = dict()
TRI_GRAM_COUNT = dict()
FOUR_GRAM_COUNT = dict()


def get_n_grams(sentence, n):
    """
    Return the ngrams including starting and stop words generated from a sentence.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    :param sentence: list of words, words are string
    :param n: 2-gram , 3-gram or 4-gram
    :return: ngrams generated from a sentence
    """
    return nltk.ngrams(sentence, n, pad_left=True, left_pad_symbol=START_SYMBOL)


def katz_back_off():
    return None


def calculate_n_grams_probability(n):
    """
    returns a dict having probability for each n-gram

    :param n: n can be 1 , 2, 3 or 4
    :return: return a dict having probability for each n-gram
    """
    # Count the total number of words in corpus, without START_SYMBOL but include STOP_SYMBOL
    total_vocab_words = 0
    n_gram_p = dict()
    if n == 1:
        for uni_gram in UNI_GRAM_COUNT:
            total_vocab_words += UNI_GRAM_COUNT[uni_gram]

        log_n = math.log(n, 2)
        for uni_gram in UNI_GRAM_COUNT:
            n_gram_p[uni_gram] = math.log(UNI_GRAM_COUNT[uni_gram], 2) - log_n
    elif n == 2:
        for bi_gram in BI_GRAM_COUNT:
            n_gram_p[bi_gram] = math.log(BI_GRAM_COUNT[bi_gram], 2) - math.log(UNI_GRAM_COUNT[bi_gram[0]], 2)

    elif n == 3:
        for trigram in TRI_GRAM_COUNT:
            n_gram_p[trigram] = math.log(TRI_GRAM_COUNT[trigram], 2) - math.log(TRI_GRAM_COUNT[trigram[:2]], 2)

    elif n == 4:
        for four_gram in FOUR_GRAM_COUNT:
            n_gram_p[four_gram] = math.log(FOUR_GRAM_COUNT[four_gram], 2) - math.log(FOUR_GRAM_COUNT[four_gram[:3]], 2)

    return n_gram_p


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

    # calculating counts for different n grams
    for sentence in training_data:

        sentence.append(STOP_SYMBOL)

        for token in sentence:
            if token in UNI_GRAM_COUNT:
                UNI_GRAM_COUNT[token] = UNI_GRAM_COUNT[token] + 1
            else:
                UNI_GRAM_COUNT[token] = 1

        for bi_gram in get_n_grams(sentence, 2):
            if bi_gram in BI_GRAM_COUNT:
                BI_GRAM_COUNT[bi_gram] = BI_GRAM_COUNT[bi_gram] + 1
            else:
                BI_GRAM_COUNT[bi_gram] = 1

        for trigram in get_n_grams(sentence, 3):
            if trigram in TRI_GRAM_COUNT:
                TRI_GRAM_COUNT[trigram] = TRI_GRAM_COUNT[trigram] + 1
            else:
                TRI_GRAM_COUNT[trigram] = 1

        for four_gram in get_n_grams(sentence, 4):
            if four_gram in FOUR_GRAM_COUNT:
                FOUR_GRAM_COUNT[four_gram] = FOUR_GRAM_COUNT[four_gram] + 1
            else:
                FOUR_GRAM_COUNT[four_gram] = 1


def main(data_set):
    """
    main function to implement language models. bi-gram, tri-gram , four-gram

    :param data_set: either BROWN_DATA_SET or GUTENBERG_DATA_SET
    :return:
    """
    sentences = list()

    if data_set == BROWN_DATA_SET:
        sentences = brown.sents(categories=brown.categories())
    elif data_set == GUTENBERG_DATA_SET:
        sentences = gutenberg.sents(gutenberg.fileids())

    # dividing data into training and test
    shuffle(list(sentences))
    sentences_count = len(sentences)
    train_sentences = sentences[:int(sentences_count * .8)]
    test_sentences = sentences[int(sentences_count * .8):]

    # calculate count of different n-grams
    calculate_n_grams_count(train_sentences)


if __name__ == '__main__':
    main(BROWN_DATA_SET)
