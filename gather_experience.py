import numpy as np
import gym
from ale_py import ALEInterface
from pynput import keyboard
from gym.utils.play import *

k=None

def on_press(key):
    try:
        print('key {0} pressed'.format(key.vk))
        k=key.vk
    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_release(key):
    
    if key == keyboard.Key.esc:
        # Stop listener
        return False
    else:
        k=None

if __name__ == '__main__':

    done=False
    #env = gym.make('Breakout-v0', render_mode='human')
    play(env = gym.make('Breakout-v0'))
    state=env.reset()
    s_prime=state
    #print('action_space: {}'.format(env.action_space.n))

    print(env.get_keys_to_action())
    print(env.get_action_meanings())

    action_map=env.get_keys_to_action()
    experience=[]

    print(action_map[(None,)])

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    #ale=ALEInterface()
    

    while not done:
        #probably should delete this line
        action = env.action_space.sample() 
        #print('action: {}'.format(action))

        #take action and collect reward and s'
        #s_prime, r, done, info = env.step(action) 
        s_prime, r, done, info = env.step(action) 

        #capture experience
        experience.append((state, action, s_prime))


        #set state to s_prime
        state=s_prime
