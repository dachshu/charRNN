import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, help='number of RNN layers', default=3)
parser.add_argument('--hidden_state', type=int, help='size of hidden state', default=1024)
parser.add_argument('--seq_length', type=int, help='length of sequence', default=10)
parser.add_argument('--learning_rate', type=float, help='set RNN learning rate', default=0.001)
parser.add_argument('--dropout_rate', type=float, help='set RNN dropout rate', default=0.5)
parser.add_argument('--batch_size', type=int, help='set RNN batch size', default=1)
parser.add_argument('file_path', type=argparse.FileType('r'), help='path of input file', metavar='FILE_PATH')
args = parser.parse_args()
print(args)
