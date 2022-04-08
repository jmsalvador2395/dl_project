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
			  trgt_model, device):

		#sample minibatch
		minibatch=random.sample(memories, batch_size)

		#split tuples into groups and convert to arrays
		states =		np.array([i[0] for i in minibatch])
		actions =		np.array([i[1] for i in minibatch])
		rewards =		np.array([i[2] for i in minibatch])
		next_states =	np.array([i[3] for i in minibatch])
		done =			np.array([i[4] for i in minibatch])

		#convert arrays to torch tensors
		states =		torch.tensor(states).to(device)
		actions =		torch.tensor(actions).to(device)
		rewards =		torch.tensor(rewards).to(device)
		next_states =	torch.tensor(next_states).to(device)
		done =			torch.tensor(done).to(device)




		pass

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

	def select_action(self, state, eps=0):
		sample=random.random()

if __name__ == '__main__':
	episodes=300
	batch_size=400
	gamma=.999
	epsilon=.9
	epsilon_start=.9
	epsilon_end=.05
	epsilon_decay=200
	target_update=10
	dtype=torch.float32

	memory_size=4000

	env = gym.make('Breakout-v0', obs_type='grayscale', render_mode='human')

	'''
	print(env.get_keys_to_action())
	print(env.get_action_meanings())
	'''

	action_map=env.get_keys_to_action()
	A=env.action_space.n

	#s=utilities.data_point()
	policy_net=dqn().to(device)
	policy_net.eval()
	trgt_policy_net=dqn().to(device)
	#target_net.load_state_dict(policy_net.state_dict())
	#target_net.eval()

	optimizer=optim.RMSprop(policy_net.parameters())
	#memory=replay_memory(10000)

	replay_memories=[]

	steps_done=0

	done=False
	for ep in range(episodes):
		print('episode {}'.format(ep))
		s_builder=data_point()
		s_builder.add_frame(env.reset())
		s=s_builder.get()
		#state=env.reset()
		t=1
		done=False
		while not done:
			
			if np.random.uniform(0, 1) < epsilon:
				a=np.random.randint(0, A)
			else:
				with torch.no_grad():
					q_vals=policy_net(torch.tensor(s[np.newaxis], dtype=dtype, device=device))
					a=int(torch.argmax(q_vals[0]))

			#take action and collect reward and s'
			s_prime_frame, r, done, info = env.step(a) 
			s_builder.add_frame(s_prime_frame)
			s_prime=s_builder.get()

			#append as (s, a, r, s', done)
			replay_memories.append((s.copy(), a, r, s_prime.copy(), done))
			if len(replay_memories) > memory_size:
				print('pruned memories')
				replay_memories.pop(0)

			s=s_prime

			#perform gradient descent step
			if len(replay_memories) >= batch_size:
				policy_net.update_model(replay_memories, batch_size, gamma, 
								 trgt_policy_net, device)

			#TODO set target weights



			t+=1
				
				



