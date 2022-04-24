import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':
	
	vanilla=pickle.load(open('../metric_data/session_0/dqn_rewards.pickle', 'rb'))
	trex=pickle.load(open('../metric_data/session_1/dqn_rewards.pickle', 'rb'))

	data_len=min(len(vanilla['clip']), len(trex['clip']))

	x=range(500, 500*(data_len+1), 500)

	fig, ax=plt.subplots()
	ax.plot(x, vanilla['clip'][:data_len],'k--', label='vanilla')
	ax.plot(x, trex['clip'][:data_len], 'k:', label='t-rex')

		
	legend=ax.legend(loc='upper right', shadow=True)
	legend.get_frame().set_facecolor('C0')

	plt.xlabel('Episode')
	plt.ylabel('Average blocks broken')
	plt.title('Average blocks broken per training episode')

	plt.show()

	fig, ax=plt.subplots()
	labels=['demonstrator', '$\pi_{bc}$', '$\hat{\pi}$']
	results=[4.8, 9., trex['clip'][-1]]

	ax.bar(labels, results)

	plt.ylabel('Average blocks broken')
	plt.title('Average blocks broken for each agent')

	plt.show()
	
