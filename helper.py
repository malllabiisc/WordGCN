import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json, queue, re
import gzip, queue, threading, scipy.sparse as sp
import logging, logging.config, itertools, pathlib

from pprint 	 import pprint
from threading   import Thread
from collections import defaultdict as ddict

np.set_printoptions(precision=4)

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------    
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def debug_nn(res_list, feed_dict):
	"""
	Function for debugging Tensorflow model      

	Parameters
	----------
	res_list:       List of tensors/variables to view
	feed_dict:	Feed dict required for getting values
	
	Returns
	-------
	Returns the list of values of given tensors/variables after execution

	"""
	import tensorflow as tf
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	return res

def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def getChunks(inp_list, chunk_size):
	"""
	Splits inp_list into lists of size chunk_size

	Parameters
	----------
	inp_list:       List to be splittted
	chunk_size:     Size of each chunk required
	
	Returns
	-------
	chunks of the inp_list each of size chunk_size, last one can be smaller (leftout data)
	"""
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def read_mappings(fname):
	"""
	A helper function for reading an object to identifier mapping

	Parameters
	----------
	fname:		Name of the file containing mapping

	Returns
	-------
	mapping:	Dictionary object containing mapping information
	"""
	mapping = {}
	for line in open(fname):
		vals = line.strip().split('\t')
		if len(vals) < 2: continue
		mapping[vals[0]] = vals[1]
	return mapping

def getEmbeddings(embed_loc, wrd_list, embed_dims):
	"""
	Gives embedding for each word in wrd_list

	Parameters
	----------
	model:		Word2vec model
	wrd_list:	List of words for which embedding is required
	embed_dims:	Dimension of the embedding

	Returns
	-------
	embed_matrix:	(len(wrd_list) x embed_dims) matrix containing embedding for each word in wrd_list in the same order
	"""
	embed_list = []

	wrd2embed = {}
	for line in open(embed_loc, encoding='utf-8', errors='ignore'):
		data = line.strip().split(' ')

		# wrd, embed = data[0], data[1:]

		# Some words may be separated by space (telephone numbers, for example).
		# It's more robust to load data as follows.
		embed = data[-1*embed_dims: ] 
		wrd   = ' '.join(data[: -1*embed_dims])

		embed = list(map(float, embed))
		wrd2embed[wrd] = embed

	for wrd in wrd_list:
		if wrd in wrd2embed: 	embed_list.append(wrd2embed[wrd])
		else: 	
			print('Word not in embeddings dump {}'.format(wrd))
			embed_list.append(np.random.randn(embed_dims))

	return np.array(embed_list, dtype=np.float32)