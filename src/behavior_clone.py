import os
import pickle
import numpy as np
from utilities import data_point

data_dir='../data/'


"""
returns the data.
X is the set of data points
y is the labels
"""
def import_data():
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

def main():

	#loads all data points at once
	X, y=import_data()

	print('X is the dataset, y is the label set')
	print('X[i] corresponds to y[i]\n')
	
	#number of data points
	print('shape of X: {}'.format(X.shape))
	print('shape of y: {}'.format(y.shape))
	
	#shape of the data point
	print('shape of data point: {}'.format(X[0].shape))
	
	


if __name__ == '__main__':
	main()
