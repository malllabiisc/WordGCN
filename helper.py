import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json, queue, re
import gzip, queue, threading, scipy.sparse as sp
import logging, logging.config, itertools, pathlib

from pprint 	 import pprint
from threading   import Thread
from collections import defaultdict as ddict

np.set_printoptions(precision=4)

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def debug_nn(res_list, feed_dict):
	import tensorflow as tf
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	return res

def get_logger(name, log_dir, config_dir):
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def read_mappings(fname):
	mapping = {}
	for line in open(fname):
		vals = line.strip().split('\t')
		if len(vals) < 2: continue
		mapping[vals[0]] = vals[1]
	return mapping

def getEmbeddings(embed_loc, wrd_list, embed_dims):
	embed_list = []

	wrd2embed = {}
	for line in open(embed_loc, encoding='utf-8', errors='ignore'):
		data = line.strip().split(' ')
		wrd, embed = data[0], data[1:]
		embed = list(map(float, embed))
		wrd2embed[wrd] = embed

	for wrd in wrd_list:
		if wrd in wrd2embed: 	embed_list.append(wrd2embed[wrd])
		else: 	
			print('Word not in embeddings dump {}'.format(wrd))
			embed_list.append(np.random.randn(embed_dims))

	return np.array(embed_list, dtype=np.float32)