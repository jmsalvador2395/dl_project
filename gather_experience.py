import sys
import numpy as np
import gym
from gym.utils.play import *
import pickle
import datetime


data=[]

"""
used for saving the data after playing.
action is an integer
obs_t should be shape (210, 160, 3)
"""
def cb_routine(obs_t, obs_tp1, action, rew, done, info):
    data.append((obs_t, action))
    

if __name__ == '__main__':
    play(env = gym.make('Breakout-v0'), zoom=4, callback=cb_routine)
    with open('data/demonstrator_{}'.format(datetime.datetime.now()), 'wb') as fh:
        pickle.dump(data, fh)
