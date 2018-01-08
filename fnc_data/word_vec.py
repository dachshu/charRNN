from konlpy.tag import Komoran as Tagger
import re
import csv
import numpy as np
import gensim

class NoTrainDataException(Exception):
	def __str__(self):
		return 'there is no training data'

class NoTestDataException(Exception):
	def __str__(self):
		return 'there is no test data'

class NoNonLabeledDataException(Exception):
	def __str__(self):
		return 'there is no non labeled data'

class DataPreprocesser:
	def __init__(self, vec_file, train_file=None, test_file=None, non_labeled_file=None):
		self.tagger = Tagger()
		self.filters = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'MAG', 'MAJ', 'VX', 'VCP', 'VCN', 'SN', 'NA']
		self.tag_id = {t: i for i, t in enumerate(self.filters)}
		self.sentence_ptn = re.compile(r'(.+?)[\.\!\?][\s$]+')
		model = gensim.models.Word2Vec.load(vec_file)
		self.wv = model.wv
		del(model)

		print('load raw data')
		self.train_data = []
		self.test_data = []
		self.non_labeled_data = []

		if train_file:
			with open(train_file) as f:
				reader = csv.reader(f, delimiter='\t')
				for i, item in enumerate(reader):
					if i == 0: continue
					self.train_data.append((item[1], item[2], item[3]))
				self.train_data = np.array(self.train_data)

		if non_labeled_file:
			with open(non_labeled_file) as f:
				reader = csv.reader(f, delimiter='\t')
				for i, item in enumerate(reader):
					if i == 0: continue
					self.non_labeled_data.append((item[0], item[1], item[2]))
				self.non_labeled_data = np.array(self.non_labeled_data)

		if test_file:
			with open(test_file) as f:
				reader = csv.reader(f, delimiter='\t')
				for i, item in enumerate(reader):
					if i == 0: continue
					self.test_data.append((item[1], item[2], item[3]))
				self.test_data = np.array(self.test_data)

		elif train_file:
			test_data_rate = 0.2
			train_data_num = int(len(self.train_data)*(1-test_data_rate))
			self.test_data = self.train_data[train_data_num:]
			self.train_data = self.train_data[:train_data_num]

		print('done')


	def filtering(self, head, body):
		body_vec = []
		head_pos = [pos for pos in self.tagger.pos(head) if pos[1] in self.filters]
		if len(head_pos) == 0:
			return (None, None)

		stn_list = self.sentence_ptn.findall(body)

		for i, sen in enumerate(stn_list):
			sen_pos = [pos for pos in self.tagger.pos(sen) if pos[1] in self.filters]
			sim = len(set(sen_pos) & set(head_pos)) + ((3-i) if i < 3 else 0)
			body_vec.append((sim, sen_pos))

		result_sorted = sorted(body_vec, reverse=True, key=lambda x:x[0])
		result = []
		for i, item in enumerate(result_sorted):
			if i >= 2: break
			result += item[1]
		return (head_pos,result)

	def pos_to_word_vec(self, pos_list):
		vec_list = []
		for p in pos_list:
			if p[0] in self.wv:
				vec = self.wv[p[0]]
				vec = np.append(vec, self.tag_id[p[1]])
			else:
				vec = np.array([0]*500)
				vec = np.append(vec, self.tag_id['NA'])

			vec_list.append(vec)
		
		return vec_list

	def mini_train_batch(self, batch_size, max_seq_len):
		if len(self.train_data) == 0:
			raise NoTrainDataException

		i = 0
		total_vec = []
		label_list = []
		head_len_list = []
		total_len_list = []

		while i < batch_size:
			idx = np.random.randint(self.train_data.shape[0])
			data = self.train_data[idx]

			(head_pos, body_pos) = self.filtering(data[0], data[1])
			if head_pos and body_pos and (len(body_pos)+len(head_pos)) < max_seq_len:
				l = []
				head_vec = self.pos_to_word_vec(head_pos)
				body_vec = self.pos_to_word_vec(body_pos)
				l += head_vec
				l += body_vec
				l += [[0]*501]*(max_seq_len - len(head_vec) - len(body_vec))
				total_vec.append(l)
				label_list.append(data[2])
				head_len_list.append(len(head_vec))
				total_len_list.append(len(head_vec)+len(body_vec))

				i+=1

		return np.array(total_vec), np.reshape(np.array(label_list), [-1,1]), np.array(total_len_list), np.array(head_len_list)

	def get_test_data(self, max_seq_len):
		if len(self.test_data) == 0:
			raise NoTestDataException

		total_vec = []
		label_list = []
		head_len_list = []
		total_len_list = []
		for data in self.test_data:
			(head_pos, body_pos) = self.filtering(data[0], data[1])
			if head_pos and body_pos:
				if len(head_pos)+len(body_pos) > max_seq_len:
					body_pos = body_pos[:max_seq_len - (len(head_pos)+len(body_pos))]
				l = []
				head_vec = self.pos_to_word_vec(head_pos)
				body_vec = self.pos_to_word_vec(body_pos)
				l += head_vec
				l += body_vec
				l += [[0]*501]*(max_seq_len - len(head_vec) - len(body_vec))
				total_vec.append(l)
				label_list.append(data[2])
				head_len_list.append(len(head_vec))
				total_len_list.append(len(head_vec)+len(body_vec))
			
		return np.array(total_vec), np.reshape(np.array(label_list), [-1,1]), np.array(total_len_list), np.array(head_len_list)

	def get_non_labeled_data(self, max_seq_len):
		if len(self.non_labeled_data) == 0:
			raise NoNonLabeledDataException

		num_vec = []
		total_vec = []
		head_len_list = []
		total_len_list = []
		for data in self.non_labeled_data:
			(head_pos, body_pos) = self.filtering(data[1], data[2])
			if head_pos and body_pos:
				if len(head_pos)+len(body_pos) > max_seq_len:
					body_pos = body_pos[:max_seq_len - (len(head_pos)+len(body_pos))]
				l = []
				head_vec = self.pos_to_word_vec(head_pos)
				body_vec = self.pos_to_word_vec(body_pos)
				l += head_vec
				l += body_vec
				l += [[0]*501]*(max_seq_len - len(head_vec) - len(body_vec))
				total_vec.append(l)
				head_len_list.append(len(head_vec))
				total_len_list.append(len(head_vec)+len(body_vec))
				num_vec.append(data[0])
			
		return np.array(num_vec), np.array(total_vec), np.array(total_len_list), np.array(head_len_list)


class SampleMaker:
	def __init__(self, wv):
		self.wv = wv
		self.tagger = Tagger()

	def change_head_pos(self, head):
		if np.random.randint(10) >= 5:
			return (head, 0)

		poses = self.tagger.pos(head)
		n_words = [n for n in poses if n[1] in ['NNG', 'NNP', 'NNB', 'NP']]

		if len(n_words) > 1:
			n_words = np.array(n_words)
			np.random.shuffle(n_words)

		for w in n_words:
			if not w[0] in self.wv: continue
			sim_list = self.wv.most_similar_cosmul(w[0])
			idx = np.random.randint(5,10)
			return (head.replace(w[0], sim_list[idx][0]), 1)

		return (head, 0)


