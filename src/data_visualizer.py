import numpy as np
from utilities import import_data, visualize_block
import matplotlib.pyplot as plt

if __name__ == '__main__':

	x, y=import_data()
	choice=''
	while choice != 'q':
		idx=np.random.randint(0, len(x))
		data_point=x[idx]
		print('picked data point {}'.format(idx))
		visualize_block(data_point)
		choice=input('enter to continue, q to quit: ')


