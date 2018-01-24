import argparse
import numpy as np
import sys
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import datetime

def normalize(data):
    min_val = np.zeros(data.shape[1])
    numerator = data - min_val
    denominator = np.amax(data, 0) - min_val + 1e-7
    return numerator / (denominator), min_val, denominator

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

    if len(input_data) <= args.seq_length:
        print('too few input data')
        sys.exit(1)

    input_data = [int(time) for time in input_data]
    input_data.sort()
    data = []
    for i, time in enumerate(input_data):
        dtime = datetime.datetime.fromtimestamp(time)
        dtime = dtime.hour*60*60 + dtime.minute*60 + dtime.second
        if i == 0:
            data.append([dtime, 0])
        else:
            data.append([dtime, time - input_data[i-1]])

    data = np.array(data)
    normal_data, n, d = normalize(data)
    #normal_data = data
    num = tf.get_variable('num', initializer=tf.constant(n, dtype=tf.float32))
    denom = tf.get_variable('denom', initializer=tf.constant(d, dtype=tf.float32))

    x = normal_data
    y = normal_data

    dataX = []
    dataY = []

    for i in range(len(x) - args.seq_length):
        _x = x[i:i+args.seq_length]
        _y = y[i+args.seq_length]
        dataX.append(_x)
        dataY.append(_y)

    train_size = int(len(dataY)*0.95)
    train_x, test_x = np.array(dataX[:train_size]), np.array(dataX[train_size:])
    train_y, test_y = np.array(dataY[:train_size]), np.array(dataY[train_size:])
    out_dim = 2

    X = tf.placeholder(tf.float32, [None, args.seq_length, 2], name='X')
    Y = tf.placeholder(tf.float32, [None, out_dim], name='Y')

    def rnn_cell():
        return tf.contrib.rnn.BasicLSTMCell(num_units=args.hidden_state, state_is_tuple=True, activation=tf.tanh)

    cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(args.num_layers)], state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], out_dim, activation_fn=None)
    Y_pred = tf.identity(Y_pred, name='y_pred')

    loss = tf.reduce_mean(tf.square(Y_pred - Y))
#loss = loss + tf.reduce_sum(tf.cast(tf.less_equal(Y_pred, 0), tf.float32)*100)
    print(loss.shape)
    print(Y_pred.shape)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    save_step = 0
    for i in range(args.epoch):
        _, l = sess.run([train, loss], feed_dict={X: train_x, Y: train_y})
        print('epoch: %d, loss: %f' % (i, l))

        if i % (args.epoch//20) == 0:
            save_path = saver.save(sess, './time_save/time_model_trained', save_step)
            save_step += 1
            print('The model is saved in', save_path)

    pre_y = sess.run(Y_pred, feed_dict={X:test_x})
    plt.plot(test_y)
    plt.plot(pre_y)
    plt.show()
    pre_y = pre_y*d+n
    for py in pre_y:
        py[0] = int(py[0])
        py[1] = int(py[1])
        h = (py[0]//3600 + 9)%24
        m = py[0]%3600//60
        s = py[0]%3600%60
        print('%d:%d:%d, %d'%(h,m,s, py[1]))
