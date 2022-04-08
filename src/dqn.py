import numpy as np
import gym
import random
from gym.utils.play import *
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
#import utilities
from utilities import data_point
import copy
import math


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

model_path='../models/'

'''
Transition=namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

is_python='inline' in matplotlib.get_backend()
if is_python:
	from IPython import display
plt.ion()
'''

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

class dqn(nn.Module):
	def __init__(self):
		super(dqn, self).__init__()
		layer1 = nn.Sequential(
			nn.Conv2d(4, 16, kernel_size=8, stride=4, padding=(1, 4)),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=(0, 2)),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		fc = nn.Linear(32*6*5, 4)

		self.model = nn.Sequential(
			layer1,
			layer2,
			nn.Flatten(),
			fc
		)

	def forward(self, x):
		x = x.to(device)
		return self.model(x)

	def update_model(self, memories, batch_size, gamma, 
			  		 alpha, trgt_model, device, optimizer):
		#set model to training mode
		self.train()

		loss_fn = nn.MSELoss()

		#set torch data type
		dt=torch.float32

		#sample minibatch
		minibatch=random.sample(memories, batch_size)

		#split tuples into groups and convert to arrays
		states =		np.array([i[0] for i in minibatch])
		actions =		np.array([i[1] for i in minibatch])
		rewards =		np.array([i[2] for i in minibatch])
		next_states =	np.array([i[3] for i in minibatch])
		done =			np.array([i[4] for i in minibatch])

		#get indeces for 
		finished_idx = 		np.where(done == True)
		unfinished_idx = 	np.where(done == False)

		#convert arrays to torch tensors
		states =		torch.tensor(states, dtype=dt).to(device)
		rewards =		torch.tensor(rewards, dtype=dt).to(device)
		next_states =	torch.tensor(next_states, dtype=dt).to(device)
		done =			torch.tensor(done).to(device)

		#create predictions
		policy_scores=self.forward(states)


		#create labels
		y=policy_scores.clone().detach()

		#create max(Q vals) from target policy net
		trgt_policy_scores=trgt_model(next_states)
		trgt_qvals=trgt_policy_scores.max(1)[0]

		#update labels
		y[range(len(y)), actions] = rewards + gamma*trgt_qvals*done
		
		#set model back to evaluation mode
		self.eval()



	'''
	def train(self, batch_size, gamma, policy_net, optimizer):
		if len(self.memory) < batch_size:
			return
		transitions=self.memory.sample(batch_size)
		
		batch=Transition(*zip(*transitions))

		non_final_mask=torch.tensor(tuple(map(lambda s: s is not None,
											batch.next_state)), device=device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

		state_batch=torch.cat(batch.state)
		action_batch=torch.cat(batch.action)
		reward_batch=torch.cat(batch.reward)

		state_action_values=self(state_batch).gather(1, action_batch)

		next_state_values=torch.zeros(batch_size, device=device)
		next_state_values[non_final_mask]=target_net(non_final_next_states).max(1)[0].detach()

		expected_state_action_values=(next_state_values*gamma)+reward_batch

		loss=nn.SmoothL1Loss()
		loss=criterion(state_action_values, expected_state_action_values.unsqueeze(1))

		optimizer.zero_grad()
		loss.backward()
		for param in self.parameters():
			param.grad.gdata.clamp_(-1, 1)
		optimizer.step()
		'''

def epsilon_update(epsilon, eps_start, eps_end, eps_decay, total_steps):
	return eps_end + (eps_start - eps_end) * math.exp(-1 * total_steps / eps_decay)

if __name__ == '__main__':
	
	episodes=300			#total amount of episodes to evaluate
	batch_size=400			#minibatch size for training
	gamma=.999				#gamma for MDP
	alpha=1e-2				#learning rate

	#epsilon greedy parameters
	epsilon=.9
	eps_start=.9
	eps_end=.05
	eps_decay=200			

	C=10					#update target model after every C steps
	dtype=torch.float32		#dtype for torch tensors
	total_steps=0			#tracks global time steps

	memory_size=4000		#size of replay memory buffer

	
	#create gym environment
	env = gym.make('Breakout-v0', obs_type='grayscale', render_mode='human')

	#get action space
	action_map=env.get_keys_to_action()
	A=env.action_space.n

	#initialize main network
	policy_net=dqn().to(device)
	policy_net.eval()

	#initialize target network
	trgt_policy_net=dqn().to(device)
	trgt_policy_net.load_state_dict(policy_net.state_dict())
	trgt_policy_net.eval()

	#initialize optimizer
	optimizer=optim.RMSprop(policy_net.parameters())

	#initialize some variables before getting into the main loop
	replay_memories=[]
	steps_done=0

	for ep in range(episodes):

		print('episode {}'.format(ep))

		s_builder=data_point()				#initialize phi transformation function
		s_builder.add_frame(env.reset())	

		s=s_builder.get()					#get current state

		t=1									#episodic t
		done=False							#tracks when episodes end
		while not done:
			
			#select action
			if np.random.uniform(0, 1) < epsilon:
				a=np.random.randint(0, A)
			else:
				with torch.no_grad():
					q_vals=policy_net(torch.tensor(s[np.newaxis], dtype=dtype, device=device))
					a=int(torch.argmax(q_vals[0]))

			#update epsilon value
			epsilon = epsilon_update(epsilon, eps_start, eps_end, eps_decay, total_steps)

			#take action and collect reward and s'
			s_prime_frame, r, done, info = env.step(a) 
			s_builder.add_frame(s_prime_frame)
			s_prime=s_builder.get()

			#append to replay_memories as (s, a, r, s', done)
			replay_memories.append((s.copy(), a, r, s_prime.copy(), done))
			if len(replay_memories) > memory_size:
				print('pruned memories')
				replay_memories.pop(0)

			s=s_prime

			#perform gradient descent step
			if len(replay_memories) >= batch_size:
				policy_net.update_model(replay_memories, batch_size, gamma, 
								 		alpha, trgt_policy_net, device, optimizer)

			#set target weights to policy net weights every C steps
			if total_steps % C == 0:
				trgt_policy_net.load_state_dict(policy_net.state_dict())

			#increment counters
			total_steps+=1
			t+=1
				
				



