import argparse
import numpy as np
import sys
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7), numerator, denominator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, help='number of RNN layers', default=1)
    parser.add_argument('--hidden_state', type=int, help='size of hidden state', default=100)
    parser.add_argument('--epoch', type=int, help='number of epoch', default=500)
    parser.add_argument('--seq_length', type=int, help='length of sequence', default=10)
    parser.add_argument('--learning_rate', type=float, help='set RNN learning rate', default=0.01)
    parser.add_argument('file', type=argparse.FileType('r'), help='path of input file', metavar='FILE_PATH')
    args = parser.parse_args()

    input_data = args.file.read()
    input_data = input_data.splitlines()
    input_data = [[int(time)] for time in input_data]
    input_data = np.array(input_data)
    normal_data, n, d = normalize(input_data)

    if len(input_data) <= args.seq_length:
        print('too few input data')
        sys.exit(1)

    x = normal_data
    y = normal_data

    dataX = []
    dataY = []

    for i in range(len(x) - args.seq_length):
        _x = x[i:i+args.seq_length]
        _y = y[i+args.seq_length]
        print(_x, '->', _y)
        dataX.append(_x)
        dataY.append(_y)

    train_size = int(len(dataY)*0.7)
    train_x, test_x = np.array(dataX[:train_size]), np.array(dataX[train_size:])
    train_y, test_y = np.array(dataY[:train_size]), np.array(dataY[train_size:])

    X = tf.placeholder(tf.float32, [None, args.seq_length, 1])
    Y = tf.placeholder(tf.float32, [None, 1])

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=args.hidden_state, activation=tf.tanh)
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=None)

    loss = tf.reduce_mean(tf.square(Y_pred - Y))
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(args.epoch):
        _, l = sess.run([train, loss], feed_dict={X: train_x, Y: train_y})
        print('epoch: %d, loss: %f' % (i, l))

    pre_y = sess.run(Y_pred, feed_dict={X:test_x})
    plt.plot(test_y)
    plt.plot(pre_y)
    plt.show()
