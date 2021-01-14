#%%
import numpy as np

#%%
class EvoNetwork:
    def __init__(self, i, h, o, nhidden=1, activation='tanh', final_activation='softmax', initialization='0'):
        activations = ['tanh','relu', 'sigmoid', 'linear', 'softmax']
        if activation not in activations or final_activation not in activations:
            raise ValueError(f'activation must be one of {activations}')

        initializations = ['0', 'random']
        if initialization not in initializations:
            raise ValueError(f'initialization must be one of {initializations}')
        init = 0.0 if initialization=='0' else 1.0
            
        self.nhidden = nhidden
        self.activation = activation
        self.final_activation = final_activation
        if activation=='tanh':
            self.act = lambda x: np.tanh(x)
        elif activation=='relu':
            self.act = lambda x: np.maximum(0, x)
        elif activation=='sigmoid':
            self.act = lambda x: 1/(1+np.exp(-x))
        elif activation=='linear':
            self.act = lambda x: x
        elif activation=='softmax':
            self.act = self.softmax

        if final_activation=='tanh':
            self.fact = lambda x: np.tanh(x)
        elif final_activation=='relu':
            self.fact = lambda x: np.maximum(0, x)
        elif final_activation=='sigmoid':
            self.fact = lambda x: 1/(1+np.exp(-x))
        elif final_activation=='linear':
            self.fact = lambda x: x
        elif final_activation=='softmax':
            self.fact = self.softmax
            
        self.layers = [{'layer': init*np.random.randn(i+1, h), 'activation': self.act}]
        self.layers += [{'layer': init*np.random.randn(h+1, h), 'activation': self.act} for _ in range(nhidden)]
        self.layers += [{'layer': init*np.random.randn(h+1, o), 'activation': self.fact}]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer['activation'](
                    np.dot(
                            np.concatenate((np.ones((x.shape[0], 1)), x), axis=1),
                            layer['layer']
                        )
                )
        return x
    
    def forwardParticle(self, i, x):
        for j, layer in enumerate(self.layers):
            x = layer['activation'](
                    np.dot(
                            np.concatenate((np.ones((x.shape[0], 1)), x), axis=1),
                            self.jlayers[j][:,:,i]
                        )
                )
        return x
    
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
            
    def softmax(self, x):
        xm = x.max()
        lse = xm + np.log(np.exp(x-xm).sum())
        return np.exp(x-lse)
    
    def dump(self):
        return {
                'activation': self.activation,
                'final_activation': self.final_activation,
                'nhidden': self.nhidden,
                'layers': [layer['layer'].tolist() for layer in self.layers]
                }
    
    def load(self, model):
        activation = model['activation']
        final_activation = model['final_activation']
        if activation=='tanh':
            self.act = lambda x: np.tanh(x)
        elif activation=='relu':
            self.act = lambda x: np.maximum(0, x)
        elif activation=='sigmoid':
            self.act = lambda x: 1/(1+np.exp(-x))
        elif activation=='linear':
            self.act = lambda x: x
        elif activation=='softmax':
            self.act = self.softmax

        if final_activation=='tanh':
            self.fact = lambda x: np.tanh(x)
        elif final_activation=='relu':
            self.fact = lambda x: np.maximum(0, x)
        elif final_activation=='sigmoid':
            self.fact = lambda x: 1/(1+np.exp(-x))
        elif final_activation=='linear':
            self.fact = lambda x: x
        elif final_activation=='softmax':
            self.fact = self.softmax
            
        self.activation = activation
        self.final_activation = final_activation
        self.nhidden = model['nhidden']
        self.layers = [{'layer': np.array(layer), 'activation': self.act} for layer in model['layers']]
        self.layers[self.nhidden + 1]['activation'] = self.fact
    
            