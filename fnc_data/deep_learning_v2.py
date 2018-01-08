import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import word_vec
import os
import sys

if len(sys.argv) < 2:
	print('not enough arguments')
	sys.exit()

training_epochs = 1000
batch_size = 100
hidden_size = 1000
keep_rate = 0.01
learning_rate = 0.00001
momentum = 0.5
max_seq_len = 600
is_training = tf.placeholder(tf.bool)

X = tf.placeholder(tf.float32, shape=[None, None, 501])
Y = tf.placeholder(tf.float32, shape=[None, 1])
SL = tf.placeholder(tf.int32, shape=[None])
head_num = tf.placeholder(tf.int32, shape=[None])

def lstm_cell():
	cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
	return cell


multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(3)], state_is_tuple=True)

#X2 = tf.reshape(X, [1, -1, 501])
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32, sequence_length=SL)

head_idx = tf.expand_dims(head_num-1, 1)
seq_idx = tf.expand_dims(SL-1, 1)
idx_range = tf.expand_dims(tf.range(tf.shape(outputs)[0]),1)
hidx = tf.concat([idx_range, head_idx], 1)
sidx = tf.concat([idx_range, seq_idx], 1)

x1_for_fc = tf.gather_nd(outputs, hidx)
x2_for_fc = tf.gather_nd(outputs, sidx)

hidden_layer1_size = 1024
#hidden_layer2_size = 256
x_for_fc = tf.concat([x1_for_fc, x2_for_fc], 1)
output1 = tf.contrib.layers.fully_connected(x_for_fc, hidden_layer1_size)
output1 = tf.layers.dropout(output1, 1-keep_rate, training=is_training)
#output2 = tf.contrib.layers.fully_connected(output1, hidden_layer2_size)
#output2 = tf.layers.dropout(output2, 1-keep_rate, training=is_training)
output = tf.contrib.layers.fully_connected(output1, 1, activation_fn=None)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))

#train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

predicted = tf.cast(tf.sigmoid(output) > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

saver = tf.train.Saver()
sess = tf.Session()

cmd = sys.argv[1]

if cmd == 'test':
	if len(sys.argv) < 4:
		print('not enough arguments')
		sys.exit()

	wc = word_vec.DataPreprocesser('fnc_word2vec', non_labeled_file=sys.argv[2])
	saver.restore(sess, sys.argv[3])

	print('get input data')
	x_test, seq_len_test, head_len_test = wc.get_non_labeled_data(max_seq_len)
	print('done')

	output_data = []
	for i, _ in enumerate(x_test):
		feed_dict = {X: x_test[i:i+1], SL: seq_len_test[i:i+1], head_num: head_len_test[i:i+1], is_training: False}
		p_val = sess.run(predicted, feed_dict=feed_dict)
		output_data.append('data %d: predicted=%d' % (i+1, p_val))

	with open(sys.argv[2]+'.result', 'w') as f:
		f.write('\n'.join(output_data))

elif cmd == 'labeled_test':
	if len(sys.argv) < 4:
		print('not enough arguments')
		sys.exit()

	wc = word_vec.DataPreprocesser('fnc_word2vec', test_file=sys.argv[2])
	saver.restore(sess, sys.argv[3])

	print('get input data')
	x_test, y_test, seq_len_test, head_len_test = wc.get_test_data(max_seq_len)
	print('done')

	output_data = []
	for i, _ in enumerate(x_test):
		feed_dict = {X: x_test[i:i+1], Y: y_test[i:i+1], SL: seq_len_test[i:i+1], head_num: head_len_test[i:i+1], is_training: False}
		p_val, label = sess.run([predicted, Y], feed_dict=feed_dict)
		output_data.append('data %d: predicted=%d, labeled=%d' % (i+1, p_val, label))

	with open(sys.argv[2] + '.result', 'w') as f:
		f.write('\n'.join(output_data))

	test_feed = {X: x_test, Y: y_test, SL: seq_len_test, head_num: head_len_test, is_training: False}
	test_acc = sess.run(accuracy, feed_dict=test_feed)
	print('accuracy : %.5f' % test_acc)

elif cmd == 'train':
	if len(sys.argv) < 3:
		print('not enough arguments')
		sys.exit()

	if len(sys.argv) == 3:
		wc = word_vec.DataPreprocesser('fnc_word2vec', train_file=sys.argv[2])
	else:
		wc = word_vec.DataPreprocesser('fnc_word2vec', train_file=sys.argv[2], test_file=sys.argv[3])

	print('get test data')
	x_test, y_test, seq_len_test, head_len_test = wc.get_test_data(max_seq_len)
	print('done')

	sess.run(tf.global_variables_initializer())

	print('start training')
	best_acc = 0
	total_data_size = 1500
	for epoch in range(training_epochs):
		avg_cost = 0
		avg_acc = 0
		for i in range((total_data_size//batch_size)):
			bx, by, bsl, bhv = wc.mini_train_batch(batch_size, max_seq_len)
			feed_dict = {X: bx, Y: by, SL: bsl, head_num: bhv, is_training: True}
			c, _, acc = sess.run([cost, train, accuracy], feed_dict=feed_dict)
			print(acc, end=' ')
			avg_cost += c/(total_data_size//batch_size)
			avg_acc += acc/(total_data_size//batch_size)

		print('')
		test_feed = {X: x_test, Y: y_test, SL: seq_len_test, head_num: head_len_test, is_training: False}
		test_acc = sess.run(accuracy, feed_dict=test_feed)
		print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =', '{:.5}'.format(avg_acc), 'test_accuracy =', '{:.5}'.format(test_acc))
		if avg_acc > 0.5 and ((best_acc < 0.6 and test_acc >= 0.55) or test_acc >= 0.6):
			if not os.path.isdir('./trained_model'):
				os.mkdir('./trained_model')
			saver.save(sess, './trained_model/model-%d-%.2f-%.2f.ckpt' % (epoch+1, avg_acc, test_acc))
			best_acc = test_acc

	print('Learning Finished!')

	for i, _ in enumerate(x_test):
		feed_dict = {X: x_test[i:i+1], Y: y_test[i:i+1], SL: seq_len_test[i:i+1], head_num: head_len_test[i:i+1], is_training: False}
		print(sess.run([predicted, Y], feed_dict=feed_dict))
