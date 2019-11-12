from models import Model
from helper import *

import tensorflow as tf

from web.embedding import Embedding
from web.evaluate  import evaluate_on_all

class SemGCN(Model):

	def make_batch(self, shuffle = True):
		"""
		Generates batches and puts them in the queue

		Parameters
		----------
		shuffle:	Whether to shuffle batches or not

		Returns
		-------
		A batch in the form of a diciontary
			Edges:	Dependency parse edges
			Words:	Word in the batch
			Cur_len:Total number of words in each sentence
		"""
		batch = []
		self.sent_num = 0
		num_batch = 0

		for semantic in self.semantic_list:

			for line in open('./semantic_info/{}.txt'.format(semantic), encoding='utf-8', errors='ignore'):
				ele = {}
				ele['Words']   = [self.voc2id[wrd.lower()] for wrd in line.strip().split() if wrd.lower() in self.voc2id]
				random.shuffle(ele['Words'])

				if len(ele['Words']) < 2: continue

				ele['Edges']   = [[i, j, self.lbl2id[semantic]] for i, j in itertools.permutations(range(len(ele['Words'])), 2)]
				ele['Cur_len'] = len(ele['Words'])
				ele['Edges']   = [[e[0], e[1], e[2]] for e in ele['Edges'] if ele['Words'][e[0]] != 0 and ele['Words'][e[1]] != 0]
				batch.append(ele)

				if len(batch) == self.p.batch_size:
					if shuffle: random.shuffle(batch)
					self.batch_queue.put(batch)
					num_batch += 1
					batch = []

		self.batch_queue.put(None)

	def getBatches(self, shuffle = True):
		"""
		Returns a generator of batches

		Parameters
		----------
		shuffle:	Whether to shuffle batches or not

		Returns
		-------
		Batch generator
		"""
		self.read_thread = Thread(target = self.make_batch)
		self.read_thread.daemon = True
		self.read_thread.start()

		random.shuffle(self.semantic_list)

		while True:
			batch = self.batch_queue.get()
			self.sent_num += self.p.batch_size
			if batch == None: break
			else:		  yield batch

	def load_data(self):
		"""
		Loads the data 

		Parameters
		----------
		voc2id:		Mapping of word to its unique identifier
		id2voc:		Inverse of voc2id
		id2freq:	Mapping of word id to its frequency in the corpus
		wrd_list:	List of words for which embedding is required
		embed_dims:	Dimension of the embedding
		voc_size:	Total number of words in vocabulary
		wrd_list:	List of words in the vocabulary
		de2id:		Mapping of edge labels of dependency parse to unique identifier
		num_deLabel:	Number of edge types in dependency graph
		rej_prob:	Word rejection probability (frequent words are rejected with higher frequency)

		Returns
		-------
		"""
		self.logger.info("Loading data")

		self.voc2id         = read_mappings('./data/voc2id.txt'); self.voc2id  = {k: int(v) for k, v in self.voc2id.items()}
		self.id2voc 	    = {v:k for k, v in self.voc2id.items()}
		self.id2freq        = read_mappings('./data/id2freq.txt');    self.id2freq = {int(k): int(v) for k, v in self.id2freq.items()}
		self.vocab_size     = len(self.voc2id)

		corpus_size    	    = np.sum(list(self.id2freq.values()))
		rel_freq 	    = {_id: freq/corpus_size for _id, freq in self.id2freq.items()}
		self.rej_prob 	    = {_id: (1-self.p.subsample/rel_freq[_id])-np.sqrt(self.p.subsample/rel_freq[_id]) for _id in self.id2freq}
		self.voc_freq_l     = [self.id2freq[_id] for _id in range(len(self.voc2id))]
		self.batch_queue    = queue.Queue(500)

		self.semantic_list = []
		if self.p.semantic != 'none':
			if self.p.semantic == 'all': 	self.semantic_list = ['synonyms', 'antonyms', 'hyponyms', 'hypernyms']
			else: 				self.semantic_list = [self.p.semantic]

		self.lbl2id      = {}
		self.num_labels  = 0
		for sem in self.semantic_list:
			self.lbl2id[sem] = len(self.lbl2id)
			self.num_labels += 1

	def add_placeholders(self):
		"""
		Placeholders for the computational graph

		Parameters
		----------
		sent_wrds:	All words in the batch
		sent_mask:	Mask for removing padding
		adj_mat:	Adjacnecy matrix for each sentence in the batch
		num_words:	Total number of words in each sentence
		seq_len:	Maximum length of sentence in the entire batch

		Returns
		-------
		"""
		self.sent_wrds 		= tf.placeholder(tf.int32,     	shape=[self.p.batch_size, None],     	 	     	name='sent_wrds')
		self.sent_mask 		= tf.placeholder(tf.float32,   	shape=[self.p.batch_size, None],     	 	     	name='sent_mask')
		self.adj_mat   		= tf.placeholder(tf.bool,      	shape=[self.num_labels, self.p.batch_size, None, None], name='adj_ind')
		self.num_words		= tf.placeholder(tf.float32,   	shape=[self.p.batch_size],         			name='num_words')
		self.seq_len   		= tf.placeholder(tf.int32,   	shape=(), 		   				name='seq_len')

	def get_adj(self, edgeList, max_labels, max_nodes):
		"""
		Returns the adjacency matrix required for applying GCN 

		Parameters
		----------
		edgeList:	List of all edges
		max_labels:	Maximum number of edge labels in dependency parse
		max_nodes:	Maximum number of words in the batch

		Returns
		-------
		Adjacency matrix shape=[Number of dependency labels, Batch size, seq_len, seq_len]
		"""
		adj_mat  = np.zeros((max_labels, self.p.batch_size, max_nodes, max_nodes), np.bool)
		for i, edges in enumerate(edgeList):
			for j, (src, dest, lbl) in enumerate(edges):
				adj_mat [lbl, i, src, dest] = 1

		return adj_mat

	def padData(self, data, seq_len, cur_lens):
		"""
		Pads a given batch

		Parameters
		----------
		data:		List of tokenized sentences in a batch
		seq_len:	Maximum length of sentence in the batch
		cur_len:	Total number of words in each sentence in a batch

		Returns
		-------
		temp:		Padded word sequence
		mask:		Masking for padded words
		"""
		temp = np.full((len(data),  seq_len), self.vocab_size, np.int32)
		mask = np.zeros((len(data), seq_len), np.float32)

		for i, ele in enumerate(data):
			temp[i, :len(ele)] = ele[:seq_len]
			mask[i, :cur_lens[i]] = np.ones(cur_lens[i], np.float32)

		return temp, mask

	def pad_dynamic(self, Words, cur_lens):
		"""
		Pads a given batch

		Parameters
		----------
		Words:		List of tokenized sentences in a batch
		cur_len:	Total number of words in each sentence in a batch

		Returns
		-------
		Word_pad:	Padded word sequence
		Words_mask:	Masking for padded words
		seq_len:	Maximum length of sentence in the batch
		"""
		seq_len     	      = max([len(wrds) for wrds in Words])
		Words_pad, Words_mask = self.padData(Words, seq_len, cur_lens)

		return Words_pad, Words_mask, seq_len

	def create_feed_dict(self, batch):
		"""
		Creates the feed dictionary

		Parameters
		----------
		batch:		Batch as returned by getBatch generator

		Returns
		-------
		feed_dict:	Feed dictionary
		"""
		Words   = [ele['Words'] for ele in batch]
		Edges   = [ele['Edges'] for ele in batch]
		Cur_len = [ele['Cur_len'] for ele in batch]

		feed_dict = {}
		feed_dict[self.sent_wrds], feed_dict[self.sent_mask], seq_len  = self.pad_dynamic(Words, Cur_len)
		feed_dict[self.adj_mat]   = self.get_adj(Edges, self.num_labels, seq_len)
		feed_dict[self.seq_len]   = seq_len
		feed_dict[self.num_words] = np.float32([len(wrds)-1 for wrds in Words])

		return feed_dict

	def aggregate(self, inp, adj_mat):
		"""
		GCN aggregation operation

		Parameters
		----------
		inp:		Action from neighborhood nodes
		adj_mat:	Adjacency matrix

		Returns
		-------
		out:		Embedding obtained after aggregation operation
		"""
		return tf.matmul(tf.cast(adj_mat, tf.float32), inp)


	def gcnLayer(self, gcn_in, in_dim, gcn_dim, batch_size, max_nodes, max_labels, adj_mat, w_gating=True, num_layers=1, name="GCN"):
		"""
		GCN Layer Implementation

		Parameters
		----------
		gcn_in:		Input to GCN Layer
		in_dim:		Dimension of input to GCN Layer 
		gcn_dim:	Hidden state dimension of GCN
		batch_size:	Batch size
		max_nodes:	Maximum number of nodes in graph
		max_labels:	Maximum number of edge labels
		adj_ind:	Adjacency matrix indices
		adj_data:	Adjacency matrix data (all 1's)
		w_gating:	Whether to include gating in GCN
		num_layers:	Number of GCN Layers
		name 		Name of the layer (used for creating variables, keep it different for different layers)

		Returns
		-------
		out		List of output of different GCN layers with first element as input itself, i.e., [gcn_in, gcn_layer1_out, gcn_layer2_out ...]
		"""
		out = []
		out.append(gcn_in)

		for layer in range(num_layers):
			gcn_in    = out[-1]
			if len(out) > 1: in_dim = gcn_dim 		# After first iteration the in_dim = gcn_dim

			with tf.name_scope('%s-%d' % (name,layer)):

				with tf.variable_scope('Loop-name-%s_layer-%d' % (name, layer)) as scope:
					w_loop   = tf.get_variable('w_loop',  initializer=tf.eye(in_dim), trainable=False, regularizer=self.regularizer)
					inp_loop = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
					if self.p.dropout != 1.0: inp_loop  = tf.nn.dropout(inp_loop, keep_prob=self.p.dropout)
					loop_act = inp_loop

				act_sum = loop_act


				for lbl in range(max_labels):

					with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:
						w_in   = tf.get_variable('w_in',  	initializer=tf.eye(in_dim), 	trainable=True,  regularizer=self.regularizer)
						w_out  = tf.get_variable('w_out', 	initializer=tf.eye(in_dim), 	trainable=True,  regularizer=self.regularizer)
						b_in   = tf.get_variable('b_in',   	[1,      gcn_dim],		trainable=True,  initializer=tf.constant_initializer(0.0),		regularizer=self.regularizer)
						b_out  = tf.get_variable('b_out',  	[1,      gcn_dim],		trainable=True,  initializer=tf.constant_initializer(0.0),		regularizer=self.regularizer)

						if w_gating:
							w_gin   = tf.get_variable('w_gin',  [in_dim, 1], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
							b_gin   = tf.get_variable('b_gin',  [1], 		initializer=tf.constant_initializer(0.0),		regularizer=self.regularizer)
							w_gout  = tf.get_variable('w_gout', [in_dim, 1], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
							b_gout  = tf.get_variable('b_gout', [1], 		initializer=tf.constant_initializer(0.0),		regularizer=self.regularizer)



					with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):

						inp_in     = tf.tensordot(gcn_in, w_in, axes=[2,0]) + tf.expand_dims(b_in, axis=0)
						adj_matrix = tf.transpose(adj_mat[lbl], [0,2,1])

						if self.p.dropout != 1.0: 
							inp_in = tf.nn.dropout(inp_in, keep_prob=self.p.dropout)

						if w_gating:
							inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2,0]) + tf.expand_dims(b_gin, axis=0)
							inp_in  = inp_in * tf.sigmoid(inp_gin)
							in_act  = self.aggregate(inp_in, adj_matrix)
						else:
							in_act  = in_t


					act_sum += in_act

					with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_out    = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)
						adj_matrix = adj_mat[lbl]

						if self.p.dropout != 1.0: 
							inp_out = tf.nn.dropout(inp_out, keep_prob=self.p.dropout)

						if w_gating:
							inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2,0]) + tf.expand_dims(b_gout, axis=0)
							inp_out  = inp_out * tf.sigmoid(inp_gout)
							out_act  = self.aggregate(inp_gout, adj_matrix)
						else:
							out_act = out_t

						act_sum += out_act

				act_sum = act_sum / tf.reshape(3 * self.num_words, [self.p.batch_size, 1, 1])

				gcn_out = tf.nn.relu(act_sum) if layer != num_layers-1 else act_sum
				out.append(gcn_out)
		return out

	def add_model(self):
		"""
		Creates the Computational Graph

		Parameters
		----------

		Returns
		-------
		nn_out:		Logits for each bag in the batch
		"""

		with tf.variable_scope('Embed_mat'):
			embed_init   	    = getEmbeddings(self.p.embed_loc, [self.id2voc[i] for i in range(len(self.voc2id))], self.p.embed_dim)
			_wrd_embed   	    = tf.get_variable('embed_matrix', initializer=embed_init, trainable=True, regularizer=self.regularizer)
			wrd_pad             = tf.Variable(tf.zeros([1, self.p.embed_dim]), trainable=False)
			self.embed_matrix   = tf.concat([_wrd_embed, wrd_pad], axis=0)
			self.context_matrix = self.embed_matrix

		embed   = tf.nn.embedding_lookup(self.embed_matrix, self.sent_wrds)

		gcn_out = self.gcnLayer(gcn_in 		= embed, 		in_dim 	   = self.p.embed_dim, 		gcn_dim    = self.p.embed_dim,
					batch_size 	= self.p.batch_size, 	max_nodes  = self.seq_len, 		max_labels = self.num_labels,
					adj_mat 	= self.adj_mat, 	w_gating   = self.p.wGate, 		num_layers = self.p.gcn_layer, 	name = "GCN")

		nn_out = gcn_out[-1]
		return nn_out

	def add_loss_op(self, nn_out):
		"""
		Computes the loss for learning embeddings

		Parameters
		----------
		nn_out:		Logits for each bag in the batch

		Returns
		-------
		loss:		Computes loss
		"""

		target_words = tf.reshape(self.sent_wrds, [-1, 1])
		nn_out_flat  = tf.reshape(nn_out, [-1, self.p.embed_dim])

		neg_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
			true_classes	= tf.cast(target_words, tf.int64),
			num_true	= 1,
			num_sampled	= self.p.neg_samples * self.p.batch_size,
			unique		= True,
			distortion	= 0.75,
			range_max	= self.vocab_size,
			unigrams	= self.voc_freq_l
		)
		neg_ids = tf.cast(neg_ids, dtype=tf.int32)

		neg_ids = tf.reshape(neg_ids, [self.p.batch_size, self.p.neg_samples])
		neg_ids = tf.reshape(tf.tile(neg_ids, [1, self.seq_len]), [self.p.batch_size, self.seq_len, self.p.neg_samples])

		target_ind    = tf.concat([
				tf.expand_dims(self.sent_wrds, axis=2),
				neg_ids
		    	], axis=2)

		target_embed  = tf.nn.embedding_lookup(self.context_matrix, target_ind)
		target_labels = tf.concat([
					tf.ones( [self.p.batch_size, self.seq_len, 1], dtype=tf.float32),
					tf.zeros([self.p.batch_size, self.seq_len, self.p.neg_samples], dtype=tf.float32)],
				axis=2)

		pred	      = tf.reduce_sum(tf.expand_dims(nn_out, axis=2) * target_embed, axis=3)
		target_labels = tf.reshape(target_labels, [self.p.batch_size * self.seq_len, -1])
		pred 	      = tf.reshape(pred, [self.p.batch_size * self.seq_len, -1])
		total_loss    = tf.nn.softmax_cross_entropy_with_logits(labels=target_labels, logits=pred)

		masked_loss = total_loss * tf.reshape(self.sent_mask, [-1])
		loss 	    = tf.reduce_sum(masked_loss) / tf.reduce_sum(self.sent_mask)

		if self.regularizer != None:
			loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

		return loss

	def add_optimizer(self, loss, isAdam=True):
		"""
		Add optimizer for training variables

		Parameters
		----------
		loss:		Computed loss

		Returns
		-------
		train_op:	Training optimizer
		"""
		with tf.name_scope('Optimizer'):
			if isAdam:  optimizer = tf.train.AdamOptimizer(self.p.lr)
			else:       optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)

		return train_op

	def __init__(self, params):
		"""
		Constructor for the main function. Loads data and creates computation graph. 

		Parameters
		----------
		params:		Hyperparameters of the model

		Returns
		-------
		"""
		self.p = params

		if not os.path.isdir(self.p.log_dir): os.system('mkdir {}'.format(self.p.log_dir))
		if not os.path.isdir(self.p.emb_dir): os.system('mkdir {}'.format(self.p.emb_dir))
		
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p)); pprint(vars(self.p))
		self.p.batch_size = self.p.batch_size

		if self.p.l2 == 0.0:    self.regularizer = None
		else:           	self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

		self.load_data()
		self.add_placeholders()

		nn_out    = self.add_model()
		self.loss = self.add_loss_op(nn_out)

		if self.p.opt == 'adam': self.train_op = self.add_optimizer(self.loss)
		else:            self.train_op = self.add_optimizer(self.loss, isAdam=False)

		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None

	def checkpoint(self, loss, epoch, sess):
		"""
		Computes intrinsic scores for embeddings and dumps the embeddings embeddings

		Parameters
		----------
		epoch:		Current epoch number
		sess:		Tensorflow session object

		Returns
		-------
		"""
		embed_matrix, \
		context_matrix 	= sess.run([self.embed_matrix, self.context_matrix])
		voc2vec 	= {wrd: embed_matrix[wid] for wrd, wid in self.voc2id.items()}
		embedding 	= Embedding.from_dict(voc2vec)
		results		= evaluate_on_all(embedding)
		results 	= {key: round(val[0], 4) for key, val in results.items()}

		curr_int 	= np.mean(list(results.values()))
		self.logger.info('Current Score: {}'.format(curr_int))

		if curr_int > self.best_int_avg:
			if self.p.dump:
				self.logger.info("Saving embedding matrix")
				f = open('embeddings/{}'.format(self.p.name), 'w')
				for id, wrd in self.id2voc.items():
					f.write('{} {}\n'.format(wrd, ' '.join([str(round(v, 6)) for v in embed_matrix[id].tolist()])))

			self.best_int_avg = curr_int

	def run_epoch(self, sess, epoch, shuffle=True):
		"""
		Runs one epoch of training

		Parameters
		----------
		sess:		Tensorflow session object
		epoch:		Epoch number
		shuffle:	Shuffle data while before creates batches

		Returns
		-------
		loss:		Loss over the corpus
		"""
		losses = []

		st = time.time()
		for step, batch in enumerate(self.getBatches(shuffle)):

			feed = self.create_feed_dict(batch)
			loss, _= sess.run([self.loss, self.train_op], feed_dict=feed)
			losses.append(loss)

			if (step+1) % 10 == 0:
				self.logger.info('E:{} (Sents: {}/{} [{}]): Train Loss \t{:.5}\t{}\t{:.5}'.format(epoch, self.sent_num, self.p.total_sents, round(self.sent_num/self.p.total_sents * 100 , 1), np.mean(losses), self.p.name, self.best_int_avg))
				en = time.time()
				if (en-st) >= 3600:
					self.logger.info("One more hour is over")
					self.checkpoint(np.mean(losses), epoch, sess)
					st = time.time()

		return np.mean(losses)

	def fit(self, sess):
		"""
		Trains the model and finally evaluates on test

		Parameters
		----------
		sess:		Tensorflow session object

		Returns
		-------
		"""
		self.saver     = tf.train.Saver()
		save_dir       = 'checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		self.save_path = os.path.join(save_dir, 'best_int_avg')

		if self.p.restore:
			self.saver.restore(sess, self.save_path)

		self.best_int_avg  = 0.0

		for epoch in range(self.p.max_epochs):
			self.logger.info('Epoch: {}'.format(epoch))
			train_loss = self.run_epoch(sess, epoch)

			self.checkpoint(train_loss, epoch, sess)
			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Best Loss: {:.5}\n'.format(epoch, train_loss,  self.best_int_avg))


if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Retrofitting GCN')

	parser.add_argument('-gpu',      dest="gpu",            default='0',                		help='GPU to use')
	parser.add_argument('-name',     dest="name",           default='test',             		help='Name of the run')
	parser.add_argument('-embed',    dest="embed_loc",      required=True,				help='Embedding for initialization')
	parser.add_argument('-embed_dim',dest="embed_dim",      default=300,     	type=int,       help='Embedding Dimension')
	parser.add_argument('-total',    dest="total_sents",    default=64640,		type=int,      	help='Total number of sentences in file')
	parser.add_argument('-lr',       dest="lr",             default=0.001,  	type=float,     help='Learning rate')
	parser.add_argument('-batch',    dest="batch_size",     default=64,		type=int,       help='Batch size')
	parser.add_argument('-epoch',    dest="max_epochs",     default=200,      	type=int,     	help='Max epochs')
	parser.add_argument('-l2',       dest="l2",             default=0.0,    	type=float,     help='L2 regularization')
	parser.add_argument('-seed',     dest="seed",           default=1234,   	type=int,       help='Seed for randomization')
	parser.add_argument('-opt',      dest="opt",            default='adam',             		help='Optimizer to use for training')
	parser.add_argument('-neg',      dest="neg_samples",    default=200,		type=int,       help='Number of negative samples')
	parser.add_argument('-gcn_layer',dest="gcn_layer",      default=1,		type=int,       help='Number of layers in GCN over dependency tree')
	parser.add_argument('-noGate',   dest="wGate",          action='store_false',       		help='Use gating in GCN')
	parser.add_argument('-drop',     dest="dropout",        default=1.0,		type=float,     help='Dropout for full connected layer')
	parser.add_argument('-semantic', dest="semantic",       default='synonyms',             	help='Which semantic information to use')
	parser.add_argument('-dump',  	 dest="dump",       	action='store_true',        		help='Dump context and embed matrix')
	parser.add_argument('-restore',  dest="restore",        action='store_true',        		help='Restore from the previous best saved model')
	parser.add_argument('-logdir',   dest="log_dir",        default='./log/',       		help='Log directory')
	parser.add_argument('-embdir',   dest="emb_dir",        default='./embeddings/',       		help='Log directory')
	parser.add_argument('-config',   dest="config_dir",     default='./config/',        		help='Config directory')
	parser.add_argument('-subsample',dest="subsample",  	default=1e-4,      type=float,     	help='Subsampling parameter')

	args = parser.parse_args()

	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)

	model = SemGCN(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)

	print('Model Trained Successfully!!')