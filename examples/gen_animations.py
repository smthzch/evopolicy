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
env = gym.make("CartPole-v0", render_mode="rgb_array")
evo = EvoSolver(env)
evo.load('models/cartpole.json')

#%%
state = env.reset()[0]
shape = state.flatten().shape[0]
done = False

ims = []
while not done:
    ims += [env.render()]
    state = state.reshape((1,shape))
    act = evo.selectAction(state)
    state, reward, term, trun, _ = env.step(act)
    done = term or trun

env.close()
#%%
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, frames=ims, interval=40)
ani.save('gifs/cartpole.gif')

#%%
env = gym.make("LunarLanderContinuous-v2", render_mode="rgb_array")
evo = EvoSolver(env)
evo.load('models/lunarlander.json')

#%%
state = env.reset()[0]
shape = state.flatten().shape[0]
done = False

ims = []
while not done:
    ims += [env.render()]
    state = state.reshape((1,shape))
    act = evo.selectAction(state)
    state, reward, term, trun, _ = env.step(act)
    done = term or trun

env.close()
#%%
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, frames=ims, interval=40)
ani.save('gifs/lunarlander.gif')

#%%
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
evo = EvoSolver(env)
evo.load('models/walker_mvnormal300.json')

#%%
state = env.reset()[0]
shape = state.flatten().shape[0]
done = False

ims = []
while not done:
    ims += [env.render()]
    state = state.reshape((1,shape))
    act = evo.selectAction(state)
    state, reward, term, trun, _ = env.step(act)
    done = term or trun

env.close()

#%%
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, frames=ims, interval=40)
ani.save('gifs/walker300.gif')
