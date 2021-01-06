#%%
import os
import gym
import matplotlib.pyplot as plt

from solver import EvoSolver

plt.ion()
#%%
env = gym.make("CartPole-v0")

#%%
evo = EvoSolver(env, 
                nhidden=1, 
                hidden_width=6, 
                activation='tanh', #hidden layer activation functions: tanh, relu, sigmoid, linear
                selection='random') #action selection: random, max

#%%
evo.train(neps=50, #number of training episodes
          lr=1e-1, #lr is step_method=='weighted'
          sigma=1e-1, #jitter sigma
          batch_size=20, #how many trials does each particle run
          nparticles=30, 
          step_method='max', #weighted or max for particle update, seems to be domain dependent
          plot=False) #plot times every training epoch

#%%
plt.plot(evo.times)

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
