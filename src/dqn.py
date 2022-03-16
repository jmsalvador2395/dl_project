import numpy as np
import gym
import random
from gym.utils.play import *
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import utilities


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


Transition=namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

is_python='inline' in matplotlib.get_backend()
if is_python:
	from IPython import display
plt.ion()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class replay_memory(object):
	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)
	
	def push(self, *args):
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)
	
	def __len__(self):
		return len(self.memory)

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

		model = nn.Sequential(
			layer1,
			layer2,
			nn.Flatten(),
			fc
		)

	def forward(self, x):
		x = x.to(device)
		return self.model(x)

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

	def select_action(self, state, eps=0):
		sample=random.random()

if __name__ == '__main__':
	episodes=300
	batch_size=4000
	gamma=.999
	epsilon_start=.9
	epsilon_end=.05
	epsilon_decay=200
	target_update=10

	done=False
	env = gym.make('Breakout-v0', obs_type='grayscale', render_mode='human')

	print(env.get_keys_to_action())
	print(env.get_action_meanings())

	action_map=env.get_keys_to_action()
	print(env.action_space.n)
	print(action_map)
	experience=[]

	s=utilities.data_point()
	policy_net=dqn().to(device)
	trgt_policy_net=dqn().to(device)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	optimizer=optim.RMSprop(policy_net.parameters())
	memory=replay_memory(10000)

	steps_done=0

	done=False
	for ep in range(episodes):
		state=env.reset()
		while not done:
			"""

			#probably should delete this line
			action = env.action_space.sample() 

			#take action and collect reward and s'
			s_prime, r, done, info = env.step(action) 

			#capture experience
			experience.append((state, action, s_prime))

			#set state to s_prime
			state=s_prime
			"""

			if s.ready():
				print('goes here')
			else:
				action=0
				new_frame, reward, done, _=env.step(action)
				s.add_frame(new_frame)
				
				



