import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Saves model after each epoch
    print()
    print('----- "saving model after Epoch: %d' % epoch)

    model_json = model.to_json()
    with open("model{}.json".format(epoch), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("assignment2_weights_{}.h5".format(epoch))


if __name__ == '__main__':

    # taking one file of gutenberg corpus
    with open("austen-emma.txt", encoding='utf-8') as f:
        text = f.read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 8
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    print('Calculating X and Y')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # splitting the data into 90% train and 10% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

    # build the model: a single LSTM
    print('Training model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=200,
              callbacks=[print_callback])

    loss, _ = model.evaluate(x_test, y_test, batch_size=128, verbose=1)

    # print perplexity of lstm_char model
    print("Perplexity of lstm_char model is {}".format(np.exp(loss)))
