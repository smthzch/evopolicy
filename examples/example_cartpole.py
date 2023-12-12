#%%
import os
import gym
import matplotlib.pyplot as plt
import numpy as np

from evopolicy.solver import EvoSolver

np.random.seed(1)
#%%
env = gym.make("CartPole-v0")

#%%
evo = EvoSolver(env, 
                nhidden=2, 
                hidden_width=60, 
                activation='tanh', #hidden layer activation functions: tanh, relu, sigmoid, softmax, linear
                final_activation='softmax', #activation for output: tanh, relu, sigmoid, softmax, linear
                selection='random', #action selection: random, max
                initialization='0') #initialize policy net with 0 or random i.e. N(0,1)

#%%
evo.train(neps=150, #number of training episodes
          lr=1e-1, #lr is step_method=='weighted'
          sigma=1e-1, #jitter sigma
          batch_size=10, #how many trials does each particle run
          nparticles=30, 
          step_method='weighted', #weighted or max for particle update, seems to be domain dependent
          plot=False) #plot times every training epoch

#%%
plt.plot(evo.times)

#%%
plt.plot(evo.rewards)

#%%
env = gym.make("CartPole-v0", render_mode="human")
state = env.reset()[0]
shape = state.flatten().shape[0]
done = False
while not done:
    state = state.reshape((1,shape))
    act = evo.selectAction(state)
    next_state, reward, term, trun, _ = env.step(act)
    done = term or trun

env.close()
# %%
#not run
#save and load model parameters
#evo.save('models/cartpole.json')
#evo.load('models/cartpole.json')
# %%
