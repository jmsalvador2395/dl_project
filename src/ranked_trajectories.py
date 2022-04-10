import gym
from utilities      import *
import torch
import torch.nn as nn
from ale_py._ale_py import Action
import time


class ranked_traj:
    def get_ranked(self,epsilon):
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
        env = gym.make('Breakout-v0', obs_type='grayscale', render_mode='human')
        env.reset()
        rewards= []
        traj= []
        for i in range(2):
            done = False
            pt=data_point()
            pt.add_frame(env.reset())
            env.step(Action.FIRE)
            lives=5
            cur_reward=0
            states=[]
            while not done:
            
                state=torch.tensor(pt.get()[np.newaxis], device=device, dtype=dtype)
                states.append(state)
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = model(state)
                    action=torch.argmax(action[0])
                    
                print('action: {}'.format(action))
                new_state, reward, done, info = env.step(action)
                cur_reward = cur_reward+reward
                if done:
                    print("Episode: {}, Reward: {}".format(i,cur_reward))
                    rewards.append(cur_reward)
                    break
                if(lives>info["lives"]):
                    env.step(Action.FIRE)
                    lives=lives-1
                print(info["lives"])
                print(new_state.shape)
                pt.add_frame(new_state)
            traj.append(states)    
         
        env.close()
        return traj, rewards
    
      
if __name__ == '__main__':
    epsilon_val=[0.01,0.02]
    ranked_trajectories=[]
    rank_obj = ranked_traj()
    for epsilon in epsilon_val:
        traj, reward = rank_obj.get_ranked(epsilon)
        ranked_trajectories.append({"epsilon":epsilon,"trajectories":traj,"rewards":reward})
    
    print(ranked_trajectories)