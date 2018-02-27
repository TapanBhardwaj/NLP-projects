import math
import string
import time
from random import shuffle

import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg

START_SYMBOL = '<s>'
STOP_SYMBOL = '</s>'
UNK_SYMBOL = 'u_n_k'
BROWN_DATA_SET = 'brown'
GUTENBERG_DATA_SET = 'gutenberg'

S1 = 's1'  # Train: D1-Train, Test: D1-Test
S2 = 's2'  # Train: D2-Train, Test: D2-Test
S3 = 's3'  # Train: D1-Train + D2-Train, Test: D1-Test
S4 = 's4'  # Train: D1-Train + D2-Train, Test: D2-Test

# n-grams counts
UNI_GRAM_COUNT = dict()
BI_GRAM_COUNT = dict()
TRI_GRAM_COUNT = dict()
FOUR_GRAM_COUNT = dict()

# n-grams log probabilities
UNI_GRAM_LOG_PROB = dict()
BI_GRAM_LOG_PROB = dict()
TRI_GRAM_LOG_PROB = dict()
FOUR_GRAM_LOG_PROB = dict()

# set of unique words in corpus including stop word
TRAINING_VOCAB = set()
TOTAL_BI_GRAMS = 0

TOTAL_TRAIN_WORDS = 0
TOTAL_TEST_WORDS = 0


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


def set_training_vocab(training_data):
    global TRAINING_VOCAB
    for i in range(len(training_data)):

        for k in range(len(training_data[i])):
            TRAINING_VOCAB.add(training_data[i][k])


def insert_unk_training_data(training_data, n=50):
    """
    Inserting unk word in training data
    :param training_data:
    :param n:
    :return:
    """
    sorted_key_list = sorted(UNI_GRAM_COUNT, key=UNI_GRAM_COUNT.get)
    count_one = 0
    global TOTAL_BI_GRAMS

    TOTAL_BI_GRAMS = len(sorted_key_list)
    # print("Total Bigrams", TOTAL_BI_GRAMS)

    for i in range(len(sorted_key_list)):
        if UNI_GRAM_COUNT[sorted_key_list[i]] == 1:
            count_one += 1

    # print("total elements having frequency 1 are : {}".format(count_one))
    sorted_key_list = sorted_key_list[:n]

    unk_count = 0

    for i in range(n):
        unk_count += UNI_GRAM_COUNT[sorted_key_list[i]]

    UNI_GRAM_COUNT[UNK_SYMBOL] = unk_count
    TRAINING_VOCAB.add(UNK_SYMBOL, )

    for i in range(len(training_data)):

        for k in range(len(training_data[i])):
            if training_data[i][k] in sorted_key_list:
                training_data[i][k] = UNK_SYMBOL

    for key in sorted_key_list:
        del UNI_GRAM_COUNT[key]


def insert_unk_test_data(test_data):
    for i in range(len(test_data)):
        for k in range(len(test_data[i])):

            if test_data[i][k] not in TRAINING_VOCAB:
                test_data[i][k] = UNK_SYMBOL


def perplexity_uni_gram(test_data, total_test_words):
    """ returns perplexity for uni-gram, using back-off"""
    total_prob_sum = 0

    for i in range(len(test_data)):

        for uni_gram in test_data[i]:
            total_prob_sum += UNI_GRAM_LOG_PROB[uni_gram]

    perplexity = math.pow(2, -1 * (total_prob_sum / total_test_words))

    print("Total perplexity of uni_gram model is : {}".format(perplexity))


def perplexity_bi_gram(test_data, total_test_words):
    """ returns perplexity for bi-gram, using back-off"""
    total_prob_sum = 0

    for i in range(len(test_data)):

        bi_grams = get_n_grams(test_data[i], 2)

        for bi_gram in bi_grams:
            if bi_gram in BI_GRAM_LOG_PROB:
                total_prob_sum += BI_GRAM_LOG_PROB[bi_gram]
            else:
                total_prob_sum += UNI_GRAM_LOG_PROB[bi_gram[1]]

    perplexity = math.pow(2, -1 * (total_prob_sum / total_test_words))

    print("Total perplexity of bi_gram model is : {}".format(perplexity))


def perplexity_trigram(test_data, total_test_words):
    """ returns perplexity for tri-gram, using back-off"""
    total_prob_sum = 0

    for i in range(len(test_data)):

        trigrams = get_n_grams(test_data[i], 3)

        for trigram in trigrams:
            if trigram in TRI_GRAM_LOG_PROB:
                total_prob_sum += TRI_GRAM_LOG_PROB[trigram]
            elif trigram[1:] in BI_GRAM_LOG_PROB:
                total_prob_sum += BI_GRAM_LOG_PROB[trigram[1:]]
            else:
                total_prob_sum += UNI_GRAM_LOG_PROB[trigram[2]]

    perplexity = math.pow(2, -1 * (total_prob_sum / total_test_words))

    print("Total perplexity of trigram model is : {}".format(perplexity))


def perplexity_four_gram(test_data, total_test_words):
    """ returns perplexity for four-gram, using back-off """
    total_prob_sum = 0

    for i in range(len(test_data)):

        four_grams = get_n_grams(test_data[i], 4)

        for four_gram in four_grams:
            if four_gram in FOUR_GRAM_LOG_PROB:
                total_prob_sum += FOUR_GRAM_LOG_PROB[four_gram]
            elif four_gram[1:] in TRI_GRAM_LOG_PROB:
                total_prob_sum += TRI_GRAM_LOG_PROB[four_gram[1:]]
            elif four_gram[2:] in BI_GRAM_LOG_PROB:
                total_prob_sum += BI_GRAM_LOG_PROB[four_gram[2:]]
            else:
                total_prob_sum += UNI_GRAM_LOG_PROB[four_gram[3]]

    perplexity = math.pow(2, -1 * (total_prob_sum / total_test_words))

    print("Total perplexity of four_gram model is : {}".format(perplexity))


def calculate_n_grams_probability():
    """
    returns a dict having probability for each n-gram
    :return: return a dict having probability for each n-gram
    """
    global UNI_GRAM_LOG_PROB
    global BI_GRAM_LOG_PROB
    global TRI_GRAM_LOG_PROB
    global FOUR_GRAM_LOG_PROB

    # Count the total number of words in corpus, without START_SYMBOL but include STOP_SYMBOL
    log_n = math.log(TOTAL_TRAIN_WORDS, 2)
    for uni_gram in UNI_GRAM_COUNT:
        UNI_GRAM_LOG_PROB[uni_gram] = math.log(UNI_GRAM_COUNT[uni_gram], 2) - log_n

    for bi_gram in BI_GRAM_COUNT:
        BI_GRAM_LOG_PROB[bi_gram] = math.log(BI_GRAM_COUNT[bi_gram], 2) - math.log(UNI_GRAM_COUNT[bi_gram[0]], 2)

    for trigram in TRI_GRAM_COUNT:
        TRI_GRAM_LOG_PROB[trigram] = math.log(TRI_GRAM_COUNT[trigram], 2) - math.log(BI_GRAM_COUNT[trigram[:2]], 2)

    for four_gram in FOUR_GRAM_COUNT:
        FOUR_GRAM_LOG_PROB[four_gram] = math.log(FOUR_GRAM_COUNT[four_gram], 2) - math.log(
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

    insert_unk_training_data(training_data, 1)

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


def add_stop_symbol(sentences):
    """ adding STOP_SYMBOL at the end of every sentence, and removing '.' """
    for sent in sentences:
        if sent[-1] == '.':
            sent[-1] = STOP_SYMBOL
        else:
            sent.append(STOP_SYMBOL)


def split(sentences, train_size=0.8):
    """
    splits the sentences into test and train
    :param sentences:
    :param train_size:
    :return:
    """
    # converting all sentences to lower case
    # for i in range(len(sentences)):
    #     sentences[i] = list(map(lambda x: x.lower(), sentences[i]))
    # dividing data into training and test

    shuffle(sentences)
    sentences_count = len(sentences)
    train_sentences = sentences[:int(sentences_count * train_size)]

    test_sentences = sentences[int(sentences_count * train_size):]

    return train_sentences, test_sentences


def get_data(sub_task):
    """
    returns train data and test data according to sub_task

    :param sub_task:
    :return:
    """
    sentences_brown = list(brown.sents(brown.fileids()))
    sentences_gutenberg = list(gutenberg.sents(gutenberg.fileids()))

    # adding stop symbols
    add_stop_symbol(sentences_brown)
    add_stop_symbol(sentences_gutenberg)

    # get training and test data
    sentences_brown_train, sentences_brown_test = split(sentences_brown, 0.9)
    sentences_gutenberg_train, sentences_gutenberg_test = split(sentences_gutenberg, 0.9)

    if sub_task == S1:
        return sentences_brown_train, sentences_brown_test
    elif sub_task == S2:
        return sentences_gutenberg_train, sentences_gutenberg_test
    elif sub_task == S3:
        sentences_brown_train.extend(sentences_gutenberg_train)
        return sentences_brown_train, sentences_brown_test
    elif sub_task == S4:
        sentences_brown_train.extend(sentences_gutenberg_train)
        return sentences_brown_train, sentences_gutenberg_test
    else:
        print("Provide proper sub_task")
        exit(0)


def main(sub_task):
    """
    main function to implement language models. bi-gram, tri-gram , four-gram

    :param sub_task:
    :return:
    """
    global TRAINING_VOCAB
    global TOTAL_TRAIN_WORDS
    global TOTAL_TEST_WORDS
    global TOTAL_BI_GRAMS

    train_sentences, test_sentences = get_data(sub_task)

    for i in range(len(train_sentences)):
        TOTAL_TRAIN_WORDS += len(train_sentences[i])

    for i in range(len(test_sentences)):
        TOTAL_TEST_WORDS += len(test_sentences[i])

    # calculate count of different n-grams
    calculate_n_grams_count(train_sentences)

    # setting vocab
    set_training_vocab(train_sentences)

    # adding unk in place of Out of Vocab words
    insert_unk_test_data(test_sentences)

    # calculate probabilities of n-grams
    calculate_n_grams_probability()

    print("Starting SUB-TASK : {}".format(sub_task))
    print("Perplexity on SUB-TASK {} TRAIN DATA : ".format(sub_task))
    perplexity_uni_gram(train_sentences, TOTAL_TRAIN_WORDS)
    perplexity_bi_gram(train_sentences, TOTAL_TRAIN_WORDS)
    perplexity_trigram(train_sentences, TOTAL_TRAIN_WORDS)
    perplexity_four_gram(train_sentences, TOTAL_TRAIN_WORDS)

    print("\nPerplexity on SUB-TASK {} TEST DATA : ".format(sub_task))
    perplexity_uni_gram(test_sentences, TOTAL_TEST_WORDS)
    perplexity_bi_gram(test_sentences, TOTAL_TEST_WORDS)
    perplexity_trigram(test_sentences, TOTAL_TEST_WORDS)
    perplexity_four_gram(test_sentences, TOTAL_TEST_WORDS)


if __name__ == '__main__':
    """
    S1 = 's1'  # Train: D1-Train, Test: D1-Test
    S2 = 's2'  # Train: D2-Train, Test: D2-Test
    S3 = 's3'  # Train: D1-Train + D2-Train, Test: D1-Test
    S4 = 's4'  # Train: D1-Train + D2-Train, Test: D2-Test
    """
    # start timer
    time.clock()
    # use arg : S1, S2, S3, S4
    main(S2)
    print("Total time taken: " + str(time.clock()) + ' sec')
