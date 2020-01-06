from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

data_path = "/home/jules/PycharmProjects/AI_ENSSAT_s5/data/"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path

def read(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return [line[:-1] for line in f.readlines()]


def build_vocab(filename):
    data = "\0".join(read(filename))

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    letters, _ = list(zip(*count_pairs))
    letter_to_id = dict(zip(letters, range(len(letters))))

    return letter_to_id


def read_x_y(filename_x, filename_y, letter_to_id):
    data_x = read(filename_x)
    data_y = read(filename_y)
    data=[[],[]]
    for x, y in zip(data_x, data_y):
        data[0].extend([letter_to_id[letter] for letter in x if letter in letter_to_id])
        data[1].extend([letter_to_id[letter] for letter in y if letter in letter_to_id])
        diff = len(data[1]) - len(data[0])
        if diff > 0:
            data[0].extend([letter_to_id['\0'] for _ in range(diff+1)])
            data[1].append(letter_to_id['\0'])
        else:
            data[0].append(letter_to_id['\0'])
            data[1].extend([letter_to_id['\0'] for _ in range(abs(diff)+1)])
        0+0
    return data  # [letter_to_id[letter] for x, y in zip(data_x, data_y) if letter in letter_to_id]


def load_data():
    # get the data paths
    train_path_x = os.path.join(data_path, "s2.train.txt")
    train_path_y = os.path.join(data_path, "s2_out.train.txt")
    valid_path_x = os.path.join(data_path, "s2.valid.txt")
    valid_path_y = os.path.join(data_path, "s2_out.valid.txt")
    test_path_x = os.path.join(data_path, "s2.test.txt")
    test_path_y = os.path.join(data_path, "s2_out.test.txt")
    # build the complete vocabulary_size, then convert text data to list of integers
    letter_to_id = build_vocab(train_path_x)
    train_data = read_x_y(train_path_x, train_path_y, letter_to_id)
    valid_data = read_x_y(valid_path_x, valid_path_y, letter_to_id)
    test_data = read_x_y(test_path_x, test_path_y, letter_to_id)
    vocabulary_size = len(letter_to_id)
    reversed_dictionary = dict(zip(letter_to_id.values(), letter_to_id.keys()))

    print(train_data[:][:5])
    print(letter_to_id)
    print(vocabulary_size)
    #print(" ".join([(reversed_dictionary[a],reversed_dictionary[b]) for a,b in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary_size, reversed_dictionary


train_data, valid_data, test_data, vocabulary_size, reversed_dictionary = load_data()


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary_size, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary_size))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[0][self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[1][self.current_idx:self.current_idx + self.num_steps]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary_size)
                self.current_idx += self.skip_step
            yield x, y

num_steps = 30
batch_size = 20
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary_size,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary_size,
                                           skip_step=num_steps)

hidden_size = 500
use_dropout=False
model = Sequential()
model.add(Embedding(vocabulary_size, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary_size)))
model.add(Activation('softmax'))

optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print(model.summary())
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 50
if args.run_opt == 1:
    model.fit_generator(train_data_generator.generate(), len(train_data[0])//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data[0])//(batch_size*num_steps), callbacks=[checkpointer])
    # model.fit_generator(train_data_generator.generate(), 2000, num_epochs,
    #                     validation_data=valid_data_generator.generate(),
    #                     validation_steps=10)
    model.save(data_path + "/final_model.hdf5")
elif args.run_opt == 2:
    model = load_model(data_path + "model-40.hdf5")
    dummy_iters = 40
    example_training_generator = KerasBatchGenerator(train_data[0], num_steps, 1, vocabulary_size,
                                                     skip_step=1)
    print("Training data:")
    for i in range(dummy_iters):
        dummy = next(example_training_generator.generate())
    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_training_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps-1, :])
        true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)
    # test data set
    dummy_iters = 40
    example_test_generator = KerasBatchGenerator(test_data[0], num_steps, 1, vocabulary_size,
                                                     skip_step=1)
    print("Test data:")
    for i in range(dummy_iters):
        dummy = next(example_test_generator.generate())
    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_test_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps - 1, :])
        true_print_out += reversed_dictionary[test_data[num_steps + dummy_iters + i]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)





