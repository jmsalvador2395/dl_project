import numpy as np
from ale_py._ale_py import Action
import pickle
import os
import matplotlib.pyplot as plt
import copy



'''
https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/src/gym/envs/atari/environment.py

refer to the link above for mapping details
'''
action_map={
	None: 0,
	ord('w'): int(Action.UP),
	ord('a'): int(Action.LEFT),
	ord('d'): int(Action.RIGHT),
	ord('s'): int(Action.DOWN),
	ord(' '): int(Action.FIRE)
}

"""
used to stack data before it's appended to the dataset
"""
class data_point:

	#initialize data point. (4, 210, 160)=4 frames
	def __init__(self, shape=(4, 210, 160)):
		self.shape=shape
		self.point=np.zeros(self.shape)
		self.layer_count=0

		self.buffer=[]

	
	def ready(self):
		return self.layer_count==4
	
	
	#adds new frame to the data point. 
	def add_frame(self, frame):

		if self.layer_count == 4:
			#self.point=np.vstack((self.point[1:], np.expand_dims(frame, 0)))
			self.point[0:3]=self.point[1:4]
			self.point[3]=frame
		else:
			self.point[self.layer_count]=frame
			self.layer_count+=1
	def get(self):
		return copy.deepcopy(self.point)
	
	
class data_collector:
	def __init__(self):
		self.point=data_point()
		self.data=[]
		self.start_key=action_map[ord(' ')]
		self.in_progress=False



	"""
	used for saving the data after playing.

	action is an integer
	obs_t should be shape (210, 160)

	data_point class is used to shape the data point to be (4, 210, 160)
	1st dimension is now 4 because we stack the previous 3 frames alongside the current frame
	"""
	def callback(self, s, s_prime, a, r, done, info):
		#used to decide whether or not to record data point
		if done:
			self.in_progress=False

		#used to know when the game has actually started
		if a==self.start_key:
			self.in_progress=True

		self.point.add_frame(s)
		
		if self.point.ready() and self.in_progress:
			self.data.append((self.point.get(), a))

	"""
	returns the data
	"""
	def dump_data(self):
		return self.data
	
	"""
	prune if all 4 frames are the same and no action is taken
	run with debug=True to visualize all the frames that have been pruned
	it's pretty janky so you feel free to force quit
	"""
	def prune_data(self, debug=False):

		active_data=[]
		inactive_data=[]
		for i in range(len(self.data)):

			s, a=self.data[i]
			noop=(a==0)
			is_still=(s == s[0]).all() and noop

			if not is_still:
				active_data.append(i)
			else:
				inactive_data.append(i)

		#delete when done
		if debug==True:
			print('self.data size={}'.format(len(self.data)))
			print('kept data in\n{}\n'.format(active_data))
			print('*******************')
			print('pruned elements in\n{}'.format(inactive_data))
			for i in inactive_data:
				visualize_block(self.data[i][0])

		print('pruned {} inactive data points'.format(len(inactive_data)))
		self.data=[self.data[i] for i in active_data]

"""
returns the data.
X is the set of data points
y is the labels
"""
def import_data(data_dir='../data/'):
	print("----- importing data -----")
	files=os.listdir(data_dir)
	tuple_data=[]
	for f in files:
		if 'demonstrator' in f:
			print('reading {}'.format(f))
			new_tuple_data=pickle.load(open(data_dir+f, 'rb'))
			tuple_data+=new_tuple_data
	if len(tuple_data) == 0:
		print('** no data available **')
		return np.array([]), np.array([])
	print("----- finished importing data -----\n")

	#break up into data and labels
	X=np.array([i for i, j in tuple_data])
	y=np.array([j for i, j in tuple_data])
	return X, y

"""
use this to random sample a data point from the data folder and plot its contents
"""
def visualize_block(data_point=None):
	
	if data_point is None:
		x, y=import_data()
		if len(x)==0:
			print('No data to visualize')
			return
		N=len(x)
		idx=np.random.randint(0, N)
		data_point=x[idx]
		print('picked data point {} out of {}'.format(idx, N))

	#plot image
	#fig = plt.figure(figsize=(10, 7))
	fig = plt.figure()
	rows=2
	columns=2

	for i, frame in enumerate(data_point):
		fig.add_subplot(rows, columns, i+1)
		plt.imshow(frame, cmap='gray')
		plt.axis('off')
		plt.title('Frame {}'.format(i))

	plt.show()


"""
use this to random sample a data point and plot its contents
"""
def visualize_frame(frame=None):
	if frame is None:
		return

	#plot image
	plt.imshow(frame, cmap='gray')
	plt.show()


