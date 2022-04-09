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
import math
import datetime
import sys


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

model_path='../models/'

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

		fc1 = nn.Linear(32*6*5, 256)
		fc2 = nn.Linear(256, 4)

		self.model = nn.Sequential(
			layer1,
			layer2,
			nn.Flatten(),
			fc1,
			fc2
		)

	def forward(self, x):
		x = x.to(device)
		return self.model(x)

	def update_model(self, memories, batch_size, gamma, 
					 trgt_model, device, optimizer):
		#set model to training mode
		self.train()

		#loss_fn = nn.MSELoss()
		loss_fn = nn.SmoothL1Loss()		#Huber Loss

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

		loss=loss_fn(policy_scores, y)

		self.zero_grad()
		loss.backward()
		optimizer.step()
		
		#set model back to evaluation mode
		self.eval()

'''
local functions
'''

def epsilon_update(epsilon, eps_start, eps_end, eps_decay, step):
	return eps_end + (eps_start - eps_end) * math.exp(-1 * step / eps_decay)

def main(arg0, pre_trained_model=None, eps_start=.9, episodes=20000, batch_size=32):

	eps_start=float(eps_start)
	episodes=int(episodes)
	batch_size=int(batch_size)			#minibatch size for training
	
	gamma=.999				#gamma for MDP
	alpha=1e-2				#learning rate
	k=4						#fram skip number

	#epsilon greedy parameters
	epsilon=eps_start
	eps_end=.05
	eps_decay=1e6			#makes it so that decay applies over 1 million time steps
	frame_threshold=1e6		#used to update epsilon

	#update_steps=10			#update policy after every 
	C=10					#update target model after every C steps
	dtype=torch.float32		#dtype for torch tensors
	total_steps=0			#tracks global time steps
	frame_count=0			#used for updating epsilon

	memory_size=4000		#size of replay memory buffer

	
	#create gym environment
	env = gym.make('Breakout-v0', obs_type='grayscale', render_mode='human')

	#get action space
	action_map=env.get_keys_to_action()
	A=env.action_space.n

	#initialize main network
	policy_net=dqn().to(device)
	
	#load pre-trained model if specified
	if pre_trained_model is not None:
		policy_net=torch.load(model_path + pre_trained_model)
		print('loaded pre-trained model')
	else:
		policy_net=dqn()

	policy_net=policy_net.to(device)
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
		total_reward=0

		s_builder=data_point()				#initialize phi transformation function
		s_builder.add_frame(env.reset())	

		s=s_builder.get()					#get current state

		t=1									#episodic t
		done=False							#tracks when episodes end
		while not done:
			
			#select action using epsilon greedy policy
			if np.random.uniform(0, 1) < epsilon:
				a=np.random.randint(0, A)
			else:
				with torch.no_grad():
					q_vals=policy_net(torch.tensor(s[np.newaxis], dtype=dtype, device=device))
					a=int(torch.argmax(q_vals[0]))

			#update epsilon value
			epsilon = epsilon_update(epsilon, eps_start, eps_end, eps_decay, frame_count)

			#take action and collect reward and s'
			s_prime_frame, r, done, info = env.step(a) 
			s_builder.add_frame(s_prime_frame)
			s_prime=s_builder.get()

			#update frame count
			frame_count+=1

			#append to replay_memories as (s, a, r, s', done)
			replay_memories.append((s.copy(), a, r, s_prime.copy(), done))
			if len(replay_memories) > memory_size:
				replay_memories.pop(0)

			#perform gradient descent step
			#if len(replay_memories) >= batch_size and total_steps % update_steps == 0:
			if len(replay_memories) >= batch_size:
				policy_net.update_model(replay_memories, batch_size, gamma, 
										trgt_policy_net, device, optimizer)

			#set target weights to policy net weights every C steps
			if total_steps % C == 0:
				trgt_policy_net.load_state_dict(policy_net.state_dict())

			#increment counters
			total_steps+=1
			t+=1
			total_reward+=r

			#skip k frames
			for i in range(k):
				s_prime_frame, r, done, info = env.step(0)		#step with NOOP action
				total_reward+=r
				s_builder.add_frame(s_prime_frame)
				frame_count+=1

			#update state
			#s=s_prime
			s=s_builder.get()

			# save model cheeckpoint every 2000 time steps
			if total_steps % 2000  == 0:
				time=datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
				torch.save(policy_net, model_path + 'dqn_checkpoint_' + time + '.mdl')
				print('model checkpoint saved')


		print('episode: {0}, reward: {1}, epsilon: {2:.2f}, total_time: {3}, ep_length: {4}, frame_count: {5}'.format(ep, total_reward, epsilon, total_steps, t, frame_count))
				
				
if __name__ == '__main__':
	main(*sys.argv)


