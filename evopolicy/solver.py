import os
import json
import numpy as np
import matplotlib.pyplot as plt
import gym

from tqdm import tqdm

from evopolicy.network import EvoNetwork, PolypNetwork

class EvoSolver:
    def __init__(
        self, 
        env, 
        network_type='evo',
        nhidden=1, 
        hidden_width=12, 
        activation='tanh', 
        final_activation='softmax', 
        selection='max',
        initialization='0',
        nntype='mlp'):

        #check action selection and activations valiid
        selections = ['max', 'categorical', "random", "normal", "mvnormal", "dirichlet", 'identity']
        if selection not in selections:
            raise ValueError(f'selection must be one of {selections}')
        if selection=='random' and (final_activation!='mvnormal' and final_activation!='normal' and final_activation!='softmax' and final_activation!='dirichlet'):
            raise ValueError("Selection and activation mismatch. Must be random and one of [normal, mvnormal, dirichlet, softmax]")
        
        #random selection must be changed to categorical distribution or continuous
        #if final activation==softmax then categorical
        #if final_activation==normal then continuous
        selection = selection if selection!='random' \
            else 'categorical' if final_activation=='softmax' \
                else 'normal' if final_activation=='normal' \
                    else 'mvnormal' if final_activation=='mvnormal' \
                        else 'dirichlet' #if final_activation=='dirichlet'
        self.selection = selection

        #env info
        self.env = env
        if type(env.action_space)==gym.spaces.discrete.Discrete:
            self.action_space = self.env.action_space.n
        else: #box?
            self.action_space = self.env.action_space.shape[0]
        
        self.state_space = 1
        self.obs_disc = False
        if type(env.observation_space)==gym.spaces.discrete.Discrete:
            self.state_space = env.observation_space.n
            self.obs_disc = True
        else:
            #flatten observation space
            for i in range(len(self.env.observation_space.shape)):
                self.state_space *= self.env.observation_space.shape[i]
        
        #network
        network_types = ['evo', 'polyp']
        if network_type not in network_types:
            raise ValueError(f'network_type must be one of {network_types}')
        self.network_type = network_type
        if network_type == 'evo':
            self.policy_net = EvoNetwork(
                self.state_space,
                hidden_width,
                self.action_space,
                nhidden=nhidden,
                activation=activation,
                final_activation=final_activation,
                initialization=initialization,
                type=nntype
            )
        else:
            self.policy_net = PolypNetwork(
                self.state_space,
                self.action_space,
                activation=activation,
                final_activation=final_activation,
                initialization=initialization,
                type=nntype
            )
        
        self.times = []
        self.rewards = []
    
    def pathfind(self, particle=None, limit=None):
        state = self.env.reset()
        self.policy_net.reset()
        if self.obs_disc:
            si = state
            state = np.zeros(self.state_space, dtype=float)
            state[si] = 1.0
        state = state.reshape((1, self.state_space))
        self.path = dict(
                actions=[],
                rewards=[],
                time=0
            )
        
        done = False
        i = 0
        while not done:
            if i==limit:
                break
            action = self.selectAction(state, particle)
            next_state, reward, done, _ = self.env.step(action)
            self.path['actions'] += [action]
            self.path['rewards'] += [reward]
            self.path['time'] += 1
            if self.obs_disc:
                si = next_state
                next_state = np.zeros(self.state_space, dtype=float)
                next_state[si] = 1.0
            state = next_state.reshape((1, self.state_space))
            i+=1
            
    def train(self, 
              neps=100, 
              lr=1e-1, 
              sigma=1e-1, 
              batch_size=10, 
              nparticles=10,
              decay=1,
              decay_step=1e6,
              step_method='weighted',
              limit=None,
              infofile=None,
              modpath=None,
              plot=False):
        if self.network_type == 'polyp':
            step_method = 'max'
        last_survivor = 1
        trng = tqdm(range(neps))
        for i_episode in trng:
            if (i_episode+1)%decay_step == 0:
                lr *= decay 
                sigma *= decay 
            
            if self.network_type == 'polyp' and last_survivor == 0:
                self.policy_net.split()
            self.policy_net.jitter(sigma, nparticles)
            
            ep_rewards = []
            ep_times = []
            for i in range(nparticles):
                part_rewards = 0.0
                part_times = []
                for _ in range(batch_size):
                    self.pathfind(i, limit=limit)
                    part_rewards += sum(self.path['rewards'])
                    part_times += [self.path['time']]
                
                ep_rewards += [part_rewards]
                part_time = sum(part_times)/len(part_times)
                ep_times += [part_time]
                    
            self.times += [sum(ep_times)/len(ep_times)]
            self.rewards += [sum(ep_rewards)/len(ep_rewards)]
            
            last_survivor = self.policy_net.step(ep_rewards, lr, step_method)
            
            if infofile is not None:
                with open(infofile, 'w') as wrt:
                    json.dump({
                                'times': self.times,
                                'rewards': self.rewards
                                },
                                wrt)
            if plot: 
                plt.plot(self.times)
                plt.show()
                plt.pause(0.001)
                    
            trng.set_description(f'Time: {round(self.times[-1], 2)} Rewards: {round(self.rewards[-1], 2)}')
            
        if infofile is not None:
            with open(infofile, 'w') as wrt:
                json.dump({
                            'times': self.times,
                            'rewards': self.rewards
                            },
                            wrt)
        
    
    def selectAction(self, state, particle=None):
        state = state.reshape((1,-1))
        if particle is not None:
            act = self.policy_net.forwardParticle(particle, state)[0]
        else:
            act = self.policy_net.forward(state)[0]
        if self.selection=='max':
            action = act.argmax()
        elif self.selection=='categorical':
            action = np.random.choice(self.action_space, p=act)
        elif self.selection=='normal':
            act = act.reshape((2,-1))
            mu = act[0,:]
            sd = np.exp(act[1,:])
            action = np.random.normal(loc=mu, scale=sd)
        elif self.selection=='mvnormal':
            Z = np.random.randn(self.action_space)
            mu = act[0:self.action_space]
            L = np.zeros((self.action_space, self.action_space), dtype=float) #cholseky
            L[np.tril_indices(self.action_space)] = act[self.action_space:] 
            di = np.diag_indices(self.action_space)
            L[di] = np.exp(L[di])
            action = mu + np.dot(L, Z)
        elif self.selection=='dirichlet':
            action = np.random.dirichlet(act)
        elif self.selection=='identity':
            action = act
        return action

    def reset(self):
        self.policy_net.reset()
        return self.env.reset()
    
    def save(self, file_path):
        model = self.policy_net.dump()
        policy = dict(
                selection=self.selection,
                state_space=self.state_space,
                action_space=self.action_space,
                model=model
            )
        with open(file_path, 'w') as wrt:
            json.dump(policy, wrt)
            
    def load(self, file_path):
        with open(file_path, 'r') as rd:
            policy = json.load(rd)
        self.policy_net.load(policy['model'])
        self.selection = policy['selection']
        self.state_space = policy['state_space']
        self.action_space = policy['action_space']
    