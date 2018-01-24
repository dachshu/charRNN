import argparse
import numpy as np
import tensorflow as tf
import datetime
import time

def update_log(file_name='time_log'):
    cur_time = time.time()
    with open(file_name, 'r') as f:
        data = f.read()
        time_data = data.splitlines()
        time_data = time_data[1:]
        time_data.append(str(cur_time))
    with open(file_name, 'w') as f:
        f.write('\n'.join(time_data))

def get_next_remaining_seconds(file_name='time_log'):
    with open(file_name, 'r+') as f:
        input_data = f.read()
        input_data = input_data.splitlines()
        input_data = [int(time) for time in input_data]
        input_data.sort()
        input_data = input_data[-11:]
        data = []
        for i, time in enumerate(input_data):
            dtime = datetime.datetime.fromtimestamp(time)
            dtime = dtime.hour*60*60 + dtime.minute*60 + dtime.second
            if i == 0:
                data.append([dtime, 0])
            else:
                data.append([dtime, time - input_data[i-1]])

        data = np.array(data, dtype=np.float32)

        with tf.Session() as sess:
            ckpt_name = tf.train.latest_checkpoint('./time_save')
            saver = tf.train.import_meta_graph(ckpt_name + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./time_save'))

            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name('X:0')
            Y_pred = graph.get_tensor_by_name('y_pred:0')
            n = graph.get_tensor_by_name('num:0').eval()
            d = graph.get_tensor_by_name('denom:0').eval()
            normal_data = (data-n)/d
            normal_data = normal_data[-10:]
            normal_data = np.reshape(normal_data, (1,10,-1))

            pre_y = sess.run(Y_pred, feed_dict={X:normal_data})
            pre_y = pre_y*d-n
            cur_time = int(time.time())

            if (cur_time - input_data[-1])//int(pre_y[1]) >= 1:
                return 0;
            else:
                return int(pre_y[1]) - (cur_time - input_data[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r+'), help='path of input file', metavar='FILE_PATH')
    args = parser.parse_args()

    input_data = args.file.read()
    input_data = input_data.splitlines()
    input_data = [int(time) for time in input_data]
    input_data.sort()
    input_data = input_data[-11:]
    data = []
    for i, time in enumerate(input_data):
        dtime = datetime.datetime.fromtimestamp(time)
        dtime = dtime.hour*60*60 + dtime.minute*60 + dtime.second
        if i == 0:
            data.append([dtime, 0])
        else:
            data.append([dtime, time - input_data[i-1]])

    data = np.array(data, dtype=np.float32)

    with tf.Session() as sess:
        ckpt_name = tf.train.latest_checkpoint('./time_save')
        saver = tf.train.import_meta_graph(ckpt_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./time_save'))

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y_pred = graph.get_tensor_by_name('y_pred:0')
        n = graph.get_tensor_by_name('num:0').eval()
        d = graph.get_tensor_by_name('denom:0').eval()
        normal_data = (data-n)/d
        normal_data = normal_data[-10:]
        normal_data = np.reshape(normal_data, (1,10,-1))

        pre_y = sess.run(Y_pred, feed_dict={X:normal_data})
        pre_y = pre_y*d+n
        
        t = int(pre_y[0][0])
        print(t)
        h = t//3600 + 9
        m = t%3600//60
        s = t%3600%60
        print('%d:%d:%d' % (h, m, s))
