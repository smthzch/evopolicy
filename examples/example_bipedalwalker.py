#%%
import os
import gym
import matplotlib.pyplot as plt
import numpy as np

from evopolicy.solver import EvoSolver

np.random.seed(1)
#%%
env = gym.make('BipedalWalker-v3')

#%%
evo = EvoSolver(env, 
                nhidden=2, 
                hidden_width=120, 
                activation='tanh', #hidden layer activation functions: tanh, relu, sigmoid, softmax, linear
                final_activation='mvnormal', #activation for output: tanh, relu, sigmoid, softmax, linear, normal
                selection='random', #action selection: max, identity, random(type depends on final_activation)
                initialization='0') #initialize policy net with 0 or random i.e. N(0,1)

#%%
evo.train(neps=300, #number of training episodes
          lr=1e-1, #lr is step_method=='weighted'
          sigma=1e-1, #jitter sigma
          batch_size=5, #how many trials does each particle run
          nparticles=30, 
          limit=100,
          step_method='weighted', #weighted or max for particle update, seems to be domain dependent
          plot=False) #plot times every training epoch

#%%
plt.plot(evo.times)

#%%
plt.plot(evo.rewards)

#%%
env = gym.make('BipedalWalker-v3', render_mode="human")
state = env.reset()[0]
shape = state.flatten().shape[0]
done = False
for i in range(200):
    state = state.reshape((1,shape))
    act = evo.selectAction(state)
    next_state, reward, term, trun, _ = env.step(act)
    done = term or trun

env.close()
# %%
#not run
#save and load model parameters
#evo.save('models/walker_mvnormal300.json')
#evo.load('models/walker_mvnnormal300.json')
# %%
