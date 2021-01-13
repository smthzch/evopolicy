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
                activation='tanh', #hidden layer activation functions: tanh, relu, sigmoid, softmax, linear
                final_activation='tanh', #activation for output: tanh, relu, sigmoid, softmax, linear
                selection='identity', #action selection: random, max, identity
                initialization='0') #initialize policy net with 0 or n(0,1)

#%%
evo.train(neps=50, #number of training episodes (50-100 seem to work well, this cell can be ran again to train more steps)
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
