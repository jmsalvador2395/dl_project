import sys
import os
import numpy as np
import gym
from gym.utils.play import *
import pickle
import datetime

from utilities import data_point, data_collector

#globals
point=data_point()
data=[]
start_key=ord(' ')
in_progress=False

if __name__ == '__main__':

	#file parameters
	collector=data_collector()
	data_dir='../data/'
	time=datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
	fname=data_dir+'demonstrator_{}.pickle'.format(time)

	#create gym environment and collect data
	play(env = gym.make('Breakout-v0', obs_type='grayscale'), zoom=4, callback=collector.callback)

	#create folder for data
	if not os.path.isdir(data_dir): 
		os.mkdir(data_dir)

	#prune data
	collector.prune_data(debug=False)

	#save data to path
	with open(fname, 'wb') as fh:
		pickle.dump(collector.dump_data(), fh)

	print('*** data written to {} ***'.format(data_dir))

