import argparse
import mmap
import re
import numpy
import random

class BatchReader():
    def __init__(self, file, pos_list):
        self.file = file
        self.pos_list = pos_list
        self.next_index = 0

    def get_all_tweet(self):
        tw_list = []
        for (start, end) in self.pos_list:
            self.file.seek(start)
            tw_list.append(self.file.read(end-start))
        return tw_list

    def random_batch(self, size):
        if len(self.pos_list) <= size:
            tw_list = self.get_all_tweet()
            random.shuffle(tw_list)
        
        else:
            tw_list = []
            idx_list = numpy.random.choice(len(self.pos_list), size)
            for idx in idx_list:
                (start, end) = self.pos_list[idx]
                self.file.seek(start)
                tw_list.append(self.file.read(end-start))

        return tw_list

    def next_batch(self, size):
        pos_len = len(self.pos_list)
        if pos_len <= size and self.next_index == 0:
            tw_list = self.get_all_tweet()
            self.next_index = pos_len

        else:
            if pos_len - self.next_index <= size:
                tw_range = pos_len - self.next_index
            else:
                tw_range = size

            tw_list = []
            for i in range(self.next_index, self.next_index+tw_range):
                (start, end) = self.pos_list[i]
                self.file.seek(start)
                tw_list.append(self.file.read(end-start))

            self.next_index += tw_range

        return tw_list

    def reset(self):
        self.next_index = 0

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, help='number of RNN layers', default=3)
    parser.add_argument('--hidden_state', type=int, help='size of hidden state', default=1024)
    parser.add_argument('--seq_length', type=int, help='length of sequence', default=10)
    parser.add_argument('--learning_rate', type=float, help='set RNN learning rate', default=0.001)
    parser.add_argument('--dropout_rate', type=float, nargs=2, help='set RNN dropout rate', default=[0.5, 0.5], metavar=('INPUT_DROP_RATE', 'OUTPUT_DROP_RATE'))
    parser.add_argument('--batch_size', type=int, help='set RNN batch size', default=1)
    parser.add_argument('file', type=argparse.FileType('r'), help='path of input file', metavar='FILE_PATH')
    args = parser.parse_args()

    input_file = args.file
    input_data = input_file.read()
    
    tweet_pos = []
    for m in re.finditer(r'.+?(:?\n\n|$)', input_data, re.DOTALL):
        tweet_pos.append((m.start(), m.end()))

    return args, BatchReader(input_file, tweet_pos)

get_arg()