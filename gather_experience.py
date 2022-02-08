import numpy as np
import gym

if __name__ == '__main__':

    done=False
    env = gym.make('Breakout-v0', render_mode='human')
    state=env.reset()
    s_prime=state
    #print('action_space: {}'.format(env.action_space.n))


    experience=[]
    print(env.action_space)
    while not done:
        #probably should delete this line
        action = env.action_space.sample() 

        #take action and collect reward and s'
        s_prime, r, done, info = env.step(action) 

        #capture experience
        experience.append((state, action, s_prime))


        #set state to s_prime
        state=s_prime
