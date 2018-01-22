import argparse
import numpy as np
import tensorflow as tf
import datetime

def get_next_remaining_seconds():
    with open('time_log', 'r+') as f:
        data = f.read()
        data = data.splitlines()
        data = [[int(time)] for time in data]
        data = np.array(data)
        data = np.sort(data, 0)
        data, n, d = normalize(data)

        sess = tf.Session()
        saver = tf.train.import_meta_graph('./save/time_model_trained.cpkt.meta')
        saver.restore(sess, './save/time_model_trained.ckpt')

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y_pred = graph.get_tensor_by_name('y_pred:0')

        pre_y = sess.run(Y_pred, feed_dict={X:[normal_data[-10:]]})

def unnormalize(data, n, d):
    return data*d+n

def normalize(data):
    numerator = data - np.amin(data, 0)
    denominator = np.amax(data, 0) - np.amin(data, 0) + 1e-7
    return numerator / denominator, np.amin(data,0), denominator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r+'), help='path of input file', metavar='FILE_PATH')
    args = parser.parse_args()

    input_data = args.file.read()
    input_data = input_data.splitlines()
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

    sess = tf.Session()
    saver = tf.train.import_meta_graph('./save/time_model_trained.ckpt.meta')
    saver.restore(sess, './save/time_model_trained.ckpt')

    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    Y_pred = graph.get_tensor_by_name('y_pred:0')

    pre_y = sess.run(Y_pred, feed_dict={X:[normal_data[-10:]]})
    print(normal_data[-10:])
    print(n, d)
    print(pre_y)
    print(pre_y*d+n)
