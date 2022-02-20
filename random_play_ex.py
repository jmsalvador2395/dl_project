import numpy as np
import gym
from gym.utils.play import *

if __name__ == '__main__':

    done=False
    env = gym.make('Breakout-v0', render_mode='human')
    state=env.reset()
    s_prime=state

    print(env.get_keys_to_action())
    print(env.get_action_meanings())

    action_map=env.get_keys_to_action()
    experience=[]

    pass

    while not done:
        #probably should delete this line
        action = env.action_space.sample() 

        #take action and collect reward and s'
        s_prime, r, done, info = env.step(action) 
        print(s_prime.shape)
        #capture experience
        experience.append((state, action, s_prime))

        #set state to s_prime
        state=s_prime


