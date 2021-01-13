#%%
import os
import gym
import matplotlib.pyplot as plt
import numpy as np

from solver import EvoSolver

plt.ion()
np.random.seed(1)
#%%
env = gym.make('LunarLanderContinuous-v2')

#%%
evo = EvoSolver(env, 
                nhidden=2, 
                hidden_width=8, 
                activation='tanh', #hidden layer activation functions: tanh, relu, sigmoid, linear
                final_activation='tanh',
                selection='identity') #action selection: random, max

#%%
evo.train(neps=50, #number of training episodes
          lr=1e-1, #lr is step_method=='weighted'
          sigma=1e-1, #jitter sigma
          batch_size=10, #how many trials does each particle run
          nparticles=30, 
          step_method='max', #weighted or max for particle update, seems to be domain dependent
          plot=False) #plot times every training epoch

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
#%%
env.close()
# %%
#not run
#save and load model parameters
#evo.save('models/lunarlander.json')
#evo.load('models/lunarlander.json')
# %%