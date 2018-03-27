import numpy as np
from keras.models import model_from_json


def select_char(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    with np.errstate(divide='ignore'):
        preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def print_char(epoch):
    # Function invoked at end of each epoch. Prints generated text.
    print()

    start_index = np.random.randint(0, len(text) - maxlen - 1)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence

    output = ''
    for i in range(100):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = select_char(preds, 0.2)

        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        output = output + next_char

    print()
    print()
    print(output)
    print()
    print()


if __name__ == '__main__':
    # loading the model weights
    json_file = open('model88.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("assignment2_weights_88.h5")

    with open("austen-emma.txt", encoding='utf-8') as f:
        text = f.read().lower()

    maxlen = 8

    chars = sorted(list(set(text)))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # printing 100 character sentence
    print_char(100)
