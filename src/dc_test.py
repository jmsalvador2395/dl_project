import gym
from utilities	  import *
import torch
import torch.nn as nn
from ale_py._ale_py import Action
import time

if __name__ == '__main__':
	if torch.cuda.is_available():
		print('using GPU')
		device='cuda'
	else:
		print('using CPU')
		device='cpu'
	dtype = torch.float32
	model= torch.load("bc_model.h5")
	model.eval()
	pt= data_point() 
	env = gym.make('BreakoutDeterministic-v4', obs_type='grayscale', render_mode='human')
	env.reset()
	#print(env.action_space)
	#observation,reward,done,info = env.step(Action.FIRE)
	#print(observation.shape)
	#pt.add_frame(observation)
	
	for i in range(5):
		done = False
		pt=data_point()
		pt.add_frame(env.reset())
		env.step(Action.FIRE)
		lives=5
		while not done:
			#env.render()
			#pt.point = torch.from_numpy(pt.point)
			#state = pt.point.to(device=device, dtype=dtype)
			state=torch.tensor(pt.get()[np.newaxis], device=device, dtype=dtype)
			#state=torch.unsqueeze(state, dim=0)
			action = model(state)
			print('scores: {}'.format(action))
			action=torch.argmax(action[0])
			print('action: {}'.format(action))
			#action=env.action_space.sample()
			new_state, reward, done, info = env.step(action)
			if(lives>info["lives"]):
				env.step(Action.FIRE)
				lives=lives-1
			print(info["lives"])
			#point.point= point.point.numpy
			print(new_state.shape)
			pt.add_frame(new_state)

	  
		#observation = env.reset()
	env.close()
	
