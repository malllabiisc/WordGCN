"""
This script will work only when the ~/web_data folder has to be replaced with the data provided here:
https://drive.google.com/open?id=1VMyddIOgmkskAFN2BvI6c49Y63SHjNfF

Usage:
python switch_evaluation_data.py -split <test/valid>
"""
import os, argparse, glob

parser = argparse.ArgumentParser(description='')
parser.add_argument('-split', default='test')
args = parser.parse_args()

path_list = ['~/web_data/analogy/', '~/web_data/categorization/', '~/web_data/similarity/']

for path in path_list:
	data_names = glob.glob('{}/{}/*'.format(path, args.split))
	for full_name in data_names: 
		name = full_name.split('/')[-1]
		os.system( 'rm -r {}{}'.format(path, name))
		os.system( 'cp -r {}{}/{} {}{}'.format(path, args.split, name, path, name))