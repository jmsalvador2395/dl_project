import sys
import os
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
def cb_routine(s, s_prime, a, r, done, info):
    data.append((s, a))
    

if __name__ == '__main__':
    play(env = gym.make('Breakout-v0'), zoom=4, callback=cb_routine)
    time=datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    fldr='./data/'
    if not os.path.isdir(fldr):
        os.mkdir(fldr)
    with open(fldr+'demonstrator_{}.pickle'.format(time), 'wb') as fh:
        pickle.dump(data, fh)
