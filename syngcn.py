from models import Model
from helper import *
import tensorflow as tf, time, ctypes
from sparse import COO
from web.embedding import Embedding
from web.evaluate  import evaluate_on_all

class SynGCN(Model):

	def getBatches(self, shuffle = True):
		"""
		Returns a generator for creating batches

		Parameters
		----------
		shuffle:	Whether to shuffle batches or not

		Returns
		-------
		A batch in the form of a diciontary
			edges:	Dependency parse edges
			wrds:	Word in the batch
			negs:	List of negative samples
			sample: Subsampled words indicator
			elen:	Total number of edges in each sentence
			wlen:	Total number of words in each sentence
		"""
		self.lib.reset()
		while True:
			# max_len = 0; unused variable
			eph_ovr = self.lib.getBatch(self.edges_addr, self.wrds_addr, self.negs_addr, self.samp_addr, self.elen_addr, self.wlen_addr, 
					  	    self.p.win_size, self.p.num_neg, self.p.batch_size, ctypes.c_float(self.p.sample))
			if eph_ovr == 1: break
			yield {'edges': self.edges, 'wrds': self.wrds, 'negs': self.negs, 'sample': self.samp, 'elen': self.elen, 'wlen': self.wlen}

	def load_data(self):
		"""
		Loads the text corpus and C++ batch creation script 

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

		self.voc2id         = read_mappings('./data/voc2id.txt');   self.voc2id  = {k:      int(v) for k, v in self.voc2id.items()}
		self.id2freq        = read_mappings('./data/id2freq.txt');  self.id2freq = {int(k): int(v) for k, v in self.id2freq.items()}
		self.id2voc 	    = {v:k for k, v in self.voc2id.items()}
		self.vocab_size     = len(self.voc2id)
		self.wrd_list	    = [self.id2voc[i] for i in range(self.vocab_size)]

		self.de2id	    = read_mappings('./data/de2id.txt');    self.de2id   = {k:      int(v) for k, v in self.de2id.items()}
		self.num_deLabel    = len(self.de2id)

		# Calculating rejection probability
		corpus_size    	    = np.sum(list(self.id2freq.values()))
		rel_freq 	    = {_id: freq/corpus_size for _id, freq in self.id2freq.items()}
		self.rej_prob 	    = {_id: (1-self.p.sample/rel_freq[_id])-np.sqrt(self.p.sample/rel_freq[_id]) for _id in self.id2freq}
		self.voc_freq_l     = [self.id2freq[_id] for _id in range(len(self.voc2id))]

		if not self.p.context: self.p.win_size = 0

		self.lib = ctypes.cdll.LoadLibrary('./batchGen.so')			# Loads the C++ code for making batches
		self.lib.init()

		# Creating pointers required for creating batches
		self.edges 	= np.zeros(self.p.max_dep_len * self.p.batch_size*3, dtype=np.int32)
		self.wrds   = np.zeros(self.p.max_sent_len  * self.p.batch_size,   dtype=np.int32)
		self.samp  	= np.zeros(self.p.max_sent_len  * self.p.batch_size,   dtype=np.int32)
		self.negs  	= np.zeros(self.p.max_sent_len  * self.p.num_neg * self.p.batch_size, dtype=np.int32)
		self.wlen  	= np.zeros(self.p.batch_size, dtype=np.int32)
		self.elen  	= np.zeros(self.p.batch_size, dtype=np.int32)

		# Pointer address of above arrays
		self.edges_addr = self.edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
		self.wrds_addr = self.wrds.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
		self.negs_addr = self.negs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
		self.samp_addr = self.samp.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
		self.wlen_addr = self.wlen.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
		self.elen_addr = self.elen.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

	def add_placeholders(self):
		"""
		Placeholders for the computational graph

		Parameters
		----------
		sent_wrds:	All words in the batch
		sent_mask:	Mask for removing padding
		neg_wrds:	Negative samples
		adj_mat:	Adjacnecy matrix for each sentence in the batch
		num_words:	Total number of words in each sentence
		seq_len:	Maximum length of sentence in the entire batch

		Returns
		-------
		"""
		self.sent_wrds 		= tf.placeholder(tf.int32,     	shape=[self.p.batch_size, None],     	 	     	 name='sent_wrds')
		self.sent_mask 		= tf.placeholder(tf.float32,   	shape=[self.p.batch_size, None],     	 	     	 name='sent_mask')
		self.neg_wrds   	= tf.placeholder(tf.int32,      shape=[self.p.batch_size, None, self.p.num_neg], 	 name='neg_wrds')
		self.adj_mat   		= tf.placeholder(tf.bool,      	shape=[self.num_deLabel, self.p.batch_size, None, None], name='adj_ind')
		self.num_words		= tf.placeholder(tf.int32,   	shape=[self.p.batch_size],         			 name='num_words')
		self.seq_len   		= tf.placeholder(tf.int32,   	shape=(), 		   				 name='seq_len')

	def get_adj(self, batch, seq_len):
		"""
		Returns the adjacency matrix required for applying GCN 

		Parameters
		----------
		batch:		batch returned by getBatch generator
		seq_len:	Maximum length of sentence in the batch

		Returns
		-------
		Adjacency matrix shape=[Number of dependency labels, Batch size, seq_len, seq_len]
		"""
		num_edges = np.sum(batch['elen'])
		b_ind     = np.expand_dims(np.repeat(np.arange(self.p.batch_size), batch['elen']), axis=1)
		e_ind     = np.reshape(batch['edges'], [-1, 3])[:num_edges]

		adj_ind   = np.concatenate([b_ind, e_ind], axis=1)
		adj_ind   = adj_ind[:, [3,0,1,2]]
		adj_data  = np.ones(num_edges, dtype=np.float32)

		return COO(adj_ind.T, adj_data, shape=(self.num_deLabel, self.p.batch_size, seq_len, seq_len)).todense()

	def pad_data(self, data, dlen, sub_sample=[]):
		"""
		Pads a given batch

		Parameters
		----------
		data:		List of tokenized sentences in a batch
		dlen:		Total number of words in each sentence in a batch

		Returns
		-------
		data_pad:	Padded word sequence
		data_mask:	Masking for padded words
		max_len:	Maximum length of sentence in the batch
		"""
		max_len   = np.max(dlen)
		data_pad  = np.zeros([len(dlen), max_len], dtype=np.int32)
		data_mask = np.zeros([len(dlen), max_len], dtype=np.float32)

		offset = 0
		for i in range(len(dlen)):
			data_pad [i, :dlen[i]] = data[offset: offset + dlen[i]]
			data_mask[i, :dlen[i]] = 1
			if len(sub_sample) != 0:
				data_mask[i, :dlen[i]] *= sub_sample[offset: offset + dlen[i]]
			offset += dlen[i]

		return data_pad, data_mask, max_len


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
		feed_dict = {}
		wrds_pad, wrds_mask, seq_len 	= self.pad_data(batch['wrds'], batch['wlen'], sub_sample=batch['sample'])
		feed_dict[self.sent_wrds] 	= wrds_pad
		feed_dict[self.sent_mask] 	= wrds_mask
		feed_dict[self.seq_len]   	= seq_len
		feed_dict[self.adj_mat]   	= self.get_adj(batch, seq_len)
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

				if layer > 0 and self.p.loop:				
					with tf.variable_scope('Loop-name-%s_layer-%d' % (name, layer)) as scope:
						w_loop  = tf.get_variable('w_loop',  [in_dim, gcn_dim], initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)
						w_gloop = tf.get_variable('w_gloop', [in_dim, 1],       initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)

						inp_loop  = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
						if self.p.dropout != 1.0: inp_loop  = tf.nn.dropout(inp_loop, keep_prob=self.p.dropout)

						if w_gating:
							loop_act = tf.tensordot(gcn_in, tf.sigmoid(w_gloop), axes=[2,0])
						else:
							loop_act = inp_loop

					act_sum = loop_act
				else:
					act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])
				

				for lbl in range(max_labels):

					with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:

						w_in   = tf.get_variable('w_in',  	[in_dim, gcn_dim], 	initializer=tf.contrib.layers.xavier_initializer(), 		regularizer=self.regularizer)
						w_out  = tf.get_variable('w_out', 	[in_dim, gcn_dim], 	initializer=tf.contrib.layers.xavier_initializer(), 		regularizer=self.regularizer)
						b_in   = tf.get_variable('b_in',   	[1,      gcn_dim],	initializer=tf.constant_initializer(0.0),			regularizer=self.regularizer)
						b_out  = tf.get_variable('b_out',  	[1,      gcn_dim],	initializer=tf.constant_initializer(0.0),			regularizer=self.regularizer)

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
							in_act = self.aggregate(inp_in, adj_matrix)
							

					with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_out    = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)
						adj_matrix = adj_mat[lbl]

						if self.p.dropout != 1.0: 
							inp_out = tf.nn.dropout(inp_out, keep_prob=self.p.dropout)

						if w_gating:
							inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2,0]) + tf.expand_dims(b_gout, axis=0)
							inp_out  = inp_out * tf.sigmoid(inp_gout)
							out_act  = self.aggregate(inp_out, adj_matrix)
						else:
							out_act = self.aggregate(inp_out, adj_matrix)


					act_sum += in_act + out_act

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

			# when target embeddings for initialization is assigned
			if self.p.embed_loc: 
				embed_init		= getEmbeddings(self.p.embed_loc, [self.id2voc[i] for i in range(len(self.voc2id))], self.p.embed_dim)
				_wrd_embed 	    = tf.get_variable('embed_matrix', \
							initializer=embed_init, regularizer=self.regularizer)
			else:
				embed_init		= tf.contrib.layers.xavier_initializer()
				_wrd_embed 	    = tf.get_variable('embed_matrix',   [self.vocab_size,  self.p.embed_dim], \
							initializer=embed_init, regularizer=self.regularizer)

			wrd_pad             = tf.Variable(tf.zeros([1, self.p.embed_dim]), trainable=False)
			self.embed_matrix   = tf.concat([_wrd_embed, wrd_pad], axis=0)

			_context_matrix     = tf.get_variable('context_matrix', [self.vocab_size,  self.p.embed_dim], \
							initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)
			self.context_matrix = tf.concat([_context_matrix, wrd_pad], axis=0)
			self.context_bias   = tf.get_variable('context_bias', 	[self.vocab_size+1], 		      \
							initializer=tf.constant_initializer(0.0),           regularizer=self.regularizer)

		embed   = tf.nn.embedding_lookup(self.embed_matrix, self.sent_wrds)
		
		gcn_in     = embed
		gcn_in_dim = self.p.embed_dim

		gcn_out = self.gcnLayer(gcn_in 		= gcn_in, 		in_dim 	   = gcn_in_dim, 		gcn_dim    = self.p.embed_dim,
					batch_size 	= self.p.batch_size, 	max_nodes  = self.seq_len, 		max_labels = self.num_deLabel,
					adj_mat 	= self.adj_mat, 	num_layers = self.p.gcn_layer, 	name = "GCN")
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

		neg_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
			true_classes	= tf.cast(target_words, tf.int64),
			num_true	= 1,
			num_sampled	= self.p.num_neg * self.p.batch_size,
			unique		= True,
			distortion	= 0.75,
			range_max	= self.vocab_size,
			unigrams	= self.voc_freq_l
		)
		neg_ids = tf.cast(neg_ids, dtype=tf.int32)
		neg_ids = tf.reshape(neg_ids, [self.p.batch_size, self.p.num_neg])
		neg_ids = tf.reshape(tf.tile(neg_ids, [1, self.seq_len]), [self.p.batch_size, self.seq_len, self.p.num_neg])
		
		target_ind   = tf.concat([
				tf.expand_dims(self.sent_wrds, axis=2),
				neg_ids
		    	], axis=2)

		target_labels = tf.concat([
					tf.ones( [self.p.batch_size, self.seq_len, 1], dtype=tf.float32), 
					tf.zeros([self.p.batch_size, self.seq_len, self.p.num_neg], dtype=tf.float32)], 
				axis=2)
		target_embed  = tf.nn.embedding_lookup(self.context_matrix, target_ind)
		pred	      = tf.reduce_sum(tf.expand_dims(nn_out, axis=2) * target_embed, axis=3)
		target_labels = tf.reshape(target_labels, [self.p.batch_size * self.seq_len, -1])
		pred 	      = tf.reshape(pred, [self.p.batch_size * self.seq_len, -1])
		total_loss    = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_labels, logits=pred)

		masked_loss   = total_loss * tf.reshape(self.sent_mask, [-1])
		loss 	      = tf.reduce_sum(masked_loss) / tf.reduce_sum(self.sent_mask)

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

	def checkpoint(self, epoch, sess):
		"""
		Computes intrinsic scores for embeddings and dumps the embeddings embeddings

		Parameters
		----------
		epoch:		Current epoch number
		sess:		Tensorflow session object

		Returns
		-------
		"""
		embed_matrix, context_matrix 	= sess.run([self.embed_matrix, self.context_matrix])
		voc2vec 	= {wrd: embed_matrix[wid] for wrd, wid in self.voc2id.items()}
		embedding 	= Embedding.from_dict(voc2vec)
		results		= evaluate_on_all(embedding)
		results 	= {key: round(val[0], 4) for key, val in results.items()}
		curr_int 	= np.mean(list(results.values()))
		self.logger.info('Current Score: {}'.format(curr_int))

		if curr_int > self.best_int_avg:
			self.logger.info("Saving embedding matrix")
			f = open('{}/{}'.format(self.p.emb_dir, self.p.name), 'w')
			for id, wrd in self.id2voc.items():
				f.write('{} {}\n'.format(wrd, ' '.join([str(round(v, 6)) for v in embed_matrix[id].tolist()])))

			self.saver.save(sess=sess, save_path=self.save_path)
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
		cnt = 0

		st = time.time()
		for step, batch in enumerate(self.getBatches(shuffle)):
			feed    = self.create_feed_dict(batch)
			loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed)
			losses.append(loss)
			cnt += self.p.batch_size

			if (step+1) % 10 == 0:
				self.logger.info('E:{} (Sents: {}/{} [{}]): Train Loss \t{:.5}\t{}\t{:.5}'.format(epoch, cnt, self.p.total_sents, round(cnt/self.p.total_sents * 100 , 1), np.mean(losses), self.p.name, self.best_int_avg))
				en = time.time()
				if (en-st) >= (3600):
					self.logger.info("One more hour is over")
					self.checkpoint(epoch, sess)
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
		self.saver       = tf.train.Saver()
		save_dir  	 = 'checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		self.save_path   = os.path.join(save_dir, 'best_int_avg')

		self.best_int_avg  = 0.0

		if self.p.restore:
			self.saver.restore(sess, self.save_path)

		for epoch in range(self.p.max_epochs):
			self.logger.info('Epoch: {}'.format(epoch))
			train_loss = self.run_epoch(sess, epoch)

			self.checkpoint(epoch, sess)
			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Best Loss: {:.5}\n'.format(epoch, train_loss,  self.best_int_avg))



if __name__== "__main__":

	parser = argparse.ArgumentParser(description='WORD GCN')

	parser.add_argument('-gpu',      dest="gpu",            default='0',                	help='GPU to use')
	parser.add_argument('-name',     dest="name",           default='test_run',             help='Name of the run')
	parser.add_argument('-embed',    dest="embed_loc",      default=None,         		help='Embedding for initialization')
	parser.add_argument('-embed_dim',dest="embed_dim",      default=300,      type=int,     help='Embedding Dimension')
	parser.add_argument('-total',    dest="total_sents",    default=56974869, type=int,     help='Total number of sentences in file')
	parser.add_argument('-lr',       dest="lr",             default=0.001,    type=float,   help='Learning rate')
	parser.add_argument('-batch',    dest="batch_size",     default=128,      type=int,     help='Batch size')
	parser.add_argument('-epoch',    dest="max_epochs",     default=50,       type=int,     help='Max epochs')
	parser.add_argument('-l2',       dest="l2",             default=0.00001,  type=float,   help='L2 regularization')
	parser.add_argument('-seed',     dest="seed",           default=1234,     type=int,     help='Seed for randomization')
	parser.add_argument('-sample',	 dest="sample",	  	default=1e-4,     type=float,   help='Subsampling parameter')
	parser.add_argument('-neg',      dest="num_neg",    	default=100,      type=int,     help='Number of negative samples')
	parser.add_argument('-side_int', dest="side_int",    	default=10000,    type=int,     help='Number of negative samples')
	parser.add_argument('-gcn_layer',dest="gcn_layer",      default=1,        type=int,     help='Number of layers in GCN over dependency tree')
	parser.add_argument('-drop',     dest="dropout",        default=1.0,      type=float,   help='Dropout for full connected layer (Keep probability')
	parser.add_argument('-opt',      dest="opt",            default='adam',             	help='Optimizer to use for training')
	parser.add_argument('-dump',  	 dest="onlyDump",       action='store_true',        	help='Dump context and embed matrix')
	parser.add_argument('-context',  dest="context",        action='store_true',        	help='Include sequential context edges (default: False)')
	parser.add_argument('-restore',  dest="restore",        action='store_true',        	help='Restore from the previous best saved model')
	parser.add_argument('-embdir',   dest="emb_dir",        default='./embeddings/',       	help='Directory for storing learned embeddings')
	parser.add_argument('-logdir',   dest="log_dir",        default='./log/',       	help='Log directory')
	parser.add_argument('-config',   dest="config_dir",     default='./config/',        	help='Config directory')

	# Added these two arguments to enable others to personalize the training set. Otherwise, the programme may suffer from memory overflow easily.
	# It is suggested that the -maxlen be set no larger than 100.
	parser.add_argument('-maxsentlen',dest="max_sent_len",	default=50, 	  type=int,	help='Max length of the sentences in data.txt (default: 40)')
	parser.add_argument('-maxdeplen', dest="max_dep_len", 	default=800,	  type=int,	help='Max length of the dependency relations in data.txt (default: 800)')

	args = parser.parse_args()

	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)

	model = SynGCN(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)

	print('Model Trained Successfully!!')
