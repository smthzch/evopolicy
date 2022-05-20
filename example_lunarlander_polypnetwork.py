#%%
import os
import gym
import matplotlib.pyplot as plt
import numpy as np

from evopolicy.solver import EvoSolver

plt.ion()
np.random.seed(1)
#%%
env = gym.make('LunarLanderContinuous-v2')

#%%
evo = EvoSolver(env, 
                network_type='polyp', 
                activation='tanh', #hidden layer activation functions: tanh, relu, sigmoid, softmax, linear
                final_activation='tanh', #activation for output: tanh, relu, sigmoid, softmax, linear
                selection='identity', #action selection: random, max, identity
                initialization='0') #initialize policy net with 0 or random i.e. N(0,1)

#%%
evo.train(neps=150, #number of training episodes
          sigma=1e-1, #jitter sigma
          batch_size=10, #how many trials does each particle run
          nparticles=30,
          plot=False) #plot times every training epoch

#%%
evo.policy_net.hiddenwidth

#%%
plt.plot(evo.times)

#%%
plt.plot(evo.rewards)

#%%
state = env.reset()
shape = state.flatten().shape[0]
done = False
while not done:
    env.render()
    state = state.reshape((1,shape))
    act = evo.selectAction(state)
    state, reward, done, _ = env.step(act)

env.close()
# %%
#not run
#save and load model parameters
#evo.save('models/lunarlander.json')
#evo.load('models/lunarlander.json')
# %%
