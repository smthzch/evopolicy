#%%
import os
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from evopolicy.solver import EvoSolver

#plt.ion()
np.random.seed(1)

#%%
def animate(i):
    plt.imshow(i)
#%%
env = gym.make("CartPole-v0")
evo = EvoSolver(env)
evo.load('models/cartpole.json')

#%%
state = env.reset()
shape = state.flatten().shape[0]
done = False

ims = []
while not done:
    ims += [env.render(mode='rgb_array')]
    state = state.reshape((1,shape))
    act = evo.selectAction(state)
    state, reward, done, _ = env.step(act)

env.close()
#%%
ims = ims[::4]
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, frames=ims, interval=40)
ani.save('gifs/cartpole.gif')

#%%
env = gym.make("LunarLanderContinuous-v2")
evo = EvoSolver(env)
evo.load('models/lunarlander.json')

#%%
state = env.reset()
shape = state.flatten().shape[0]
done = False

ims = []
while not done:
    ims += [env.render(mode='rgb_array')]
    state = state.reshape((1,shape))
    act = evo.selectAction(state)
    state, reward, done, _ = env.step(act)

env.close()
#%%
ims = ims[::4]
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, frames=ims, interval=40)
ani.save('gifs/lunarlander.gif')

# %%
