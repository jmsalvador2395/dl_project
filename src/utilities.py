import numpy as np
from ale_py._ale_py import Action



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
		#return len(self.buffer)==4
	
	
	#adds new frame to the data point. 
	def add_frame(self, frame):

		if self.layer_count == 4:
			self.buffer=np.vstack((self.point[1:], np.expand_dims(frame, 0)))
		else:
			self.point[self.layer_count]=frame
			self.layer_count+=1
	
	def get(self):
		return self.point
	
	
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
	#TODO

	*** DO NOT USE THIS ***

	maybe we'll need it if training doesn't go well.
	as-is this function doesn't work though so don't
	use it until fixed

	***********************
	prune if all 4 frames are the same and no action is taken
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

		self.data=[self.data[i] for i in active_data]


