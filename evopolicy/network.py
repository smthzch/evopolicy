#%%
import numpy as np

def softmax(x):
    xm = x.max()
    lse = xm + np.log(np.exp(x-xm).sum())
    return np.exp(x-lse)

activations = {
    'tanh': lambda x: np.tanh(x),
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1/(1+np.exp(-x)),
    'linear': lambda x: x,
    'softmax': softmax,
    'normal': lambda x: x,
    'mvnormal': lambda x: x,
    'dirichlet': lambda x: np.maximum(1, x)
}

class BaseNetwork:
    def __init__(
        self, 
        h, 
        o, 
        activation='tanh', 
        final_activation='softmax', 
        initialization='0',
        type='mlp'
    ):
        if activation not in activations or final_activation not in activations:
            raise ValueError(f'activation must be one of {activations}')

        initializations = ['0', 'random']
        if initialization not in initializations:
            raise ValueError(f'initialization must be one of {initializations}')
        self.init = 0.0 if initialization=='0' else 1.0

        types = ['mlp', 'rnn']
        if type not in types:
            raise ValueError(f'type must be one of {types}')
        self.type = type
            
        self.activation = activation
        self.final_activation = final_activation
        self.act = activations[activation]
        self.fact = activations[final_activation]

        if final_activation=='normal':
            o *= 2 #return mu and log(sigma) for each output dim
        elif final_activation=='mvnormal':
            o = int(o*(o+3)/2) #return mu and log(sigma) and covariance uppertri for each output dim

        self.ih = 0
        if type=='rnn':
            self.ih = h #to concatenate hidden layer to input
            self.h = np.zeros((1,self.ih)) #hidden layer, empy if mlp
        else:
            self.h = np.zeros(0)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        if self.type=='rnn':
            x = np.concatenate((x, self.h), axis=1)
        for j, layer in enumerate(self.layers):
            x = layer['activation'](
                    np.dot(
                            np.concatenate((np.ones((x.shape[0], 1)), x), axis=1),
                            layer['layer']
                        )
                )
            if j==0 and self.type=='rnn':
                self.h = x.reshape((1,-1)).copy()
        return x
    
    def forwardParticle(self, i, x):
        x = x.reshape(x.shape[0], -1)
        if self.type=='rnn':
            x = np.concatenate((x, self.h), axis=1)
        for j, layer in enumerate(self.layers):
            x = layer['activation'](
                    np.dot(
                            np.concatenate((np.ones((x.shape[0], 1)), x), axis=1),
                            self.jlayers[j][:,:,i]
                        )
                )
            if j==0 and self.type=='rnn':
                self.h = x.reshape((1,-1)).copy()
        return x

    def reset(self):
        self.h = np.zeros_like(self.h)
    
    def jitter(self, sigma=1e-1, nparticles=10):
        self.sigma = sigma
        self.nparticles = nparticles
        self.jitters = []
        self.jlayers = []
        for layer in self.layers:
            shape = layer['layer'].shape
            jit = np.random.normal(np.zeros((shape[0], shape[1], self.nparticles)), sigma)
            jit[:,:,0] = 0
            self.jitters += [jit]
            self.jlayers += [layer['layer'][:,:,None] + jit]
            
    def step(self, R, lr, method='weighted'):
        methods = ['weighted', 'max']
        if method not in methods:
            raise ValueError(f'method must be one of {methods}')
        R = np.array(R)
        if method=='weighted':
            R = R - R.mean()
            if R.std()>0:
                R = R / R.std()
            for i, layer in enumerate(self.layers):
                layer['layer'] += lr*np.dot(self.jitters[i], R)/(self.nparticles*self.sigma)
        else: #max
            if (R==R.max()).sum()>1:
                rmaxs = np.argwhere(R==R.max())[:,0]
                rmax = np.random.choice(rmaxs)
            else:
                rmax = R.argmax()
            for i, layer in enumerate(self.layers):
                layer['layer'] += self.jitters[i][:,:,rmax]
                

#%%
class EvoNetwork(BaseNetwork):
    def __init__(
        self, 
        i, 
        h, 
        o, 
        nhidden=1, 
        activation='tanh', 
        final_activation='softmax', 
        initialization='0',
        type='mlp'):
        super().__init__(h, o, activation, final_activation, initialization, type)
        self.nhidden = nhidden
        self.hiddenwidth = h
        self.layers = [{'layer': self.init*np.random.randn(i + self.ih + 1, h), 'activation': self.act}]
        self.layers += [{'layer': self.init*np.random.randn(h + 1, h), 'activation': self.act} for _ in range(nhidden)]
        self.layers += [{'layer': self.init*np.random.randn(h + 1, o), 'activation': self.fact}]
  
                
    def dump(self):
        return {
                'activation': self.activation,
                'final_activation': self.final_activation,
                'type': self.type,
                'nhidden': self.nhidden,
                'hiddenwidth': self.hiddenwidth,
                'layers': [layer['layer'].tolist() for layer in self.layers]
                }
    
    def load(self, model):
        activation = model['activation']
        final_activation = model['final_activation']
        self.act = activations[activation]
        self.fact = activations[final_activation]

        self.type = model['type']
        self.hiddenwidth = model['hiddenwidth']
        if self.type=='rnn':
            self.h = np.zeros((1,self.hiddenwidth))
        else:
            self.h = np.zeros(0)
        self.nhidden = model['nhidden']
        self.layers = [{'layer': np.array(layer), 'activation': self.act} for layer in model['layers']]
        self.layers[self.nhidden + 1]['activation'] = self.fact
    
            
# %%
class PolypNetwork(BaseNetwork):
    def __init__(
        self, 
        i,  
        o, 
        activation='tanh', 
        final_activation='softmax', 
        initialization='0',
        type='mlp'):
        super().__init__(1, o, activation, final_activation, initialization, type)
        self.layers = [{'layer': self.init*np.random.randn(i + 1, 1), 'activation': self.act}]
        self.layers += [{'layer': self.init*np.random.randn(2, o), 'activation': self.fact}]

    def split(self):
        layer = self.layers[0]['layer']
        flayer = self.layers[1]['layer']
        split_ix = (flayer ** 2).sum(axis=1)[1:].argmax() # split node with largest norm
        # create 2 new nodes with opposite weights to cancel effect
        layer = np.concatenate(
            [layer, layer[:,[split_ix]], -layer[:,[split_ix]]],
            axis=1
        )
        self.layers[0]['layer'] = layer

        # update final node to duplicate weights
        flayer = np.concatenate(
            [flayer, flayer[[split_ix + 1],:], flayer[[split_ix + 1],:]],
            axis=0
        )
        self.layers[1]['layer'] = flayer

        self.hiddenwidth += 2
            
    def step(self, R, lr, method='max'):
        methods = ['max']
        if method not in methods:
            raise ValueError(f'method must be one of {methods}')
        R = np.array(R)
        if (R==R.max()).sum()>1:
            rmaxs = np.argwhere(R==R.max())[:,0]
            rmax = np.random.choice(rmaxs)
        else:
            rmax = R.argmax()
        for i, layer in enumerate(self.layers):
            layer['layer'] += self.jitters[i][:,:,rmax]
        return rmax
                
    def dump(self):
        return {
                'activation': self.activation,
                'final_activation': self.final_activation,
                'type': self.type,
                'hiddenwidth': self.hiddenwidth,
                'layers': [layer['layer'].tolist() for layer in self.layers]
                }
    
    def load(self, model):
        activation = model['activation']
        final_activation = model['final_activation']
        self.act = activations[activation]
        self.fact = activations[final_activation]

        self.type = model['type']
        self.hiddenwidth = model['hiddenwidth']
        if self.type=='rnn':
            self.h = np.zeros((1,self.hiddenwidth))
        else:
            self.h = np.zeros(0)
        self.layers = [{'layer': np.array(layer), 'activation': self.act} for layer in model['layers']]
        self.layers[1]['activation'] = self.fact