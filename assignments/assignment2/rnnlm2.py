from numpy import *
import itertools

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid, make_onehot
from misc import random_weight_matrix


class RNNLM(NNBase):
    """
    Language Model based on RNN
    Arguments:
        layer_dims : a tuple consisting of size of each layer
        alpha : default learning rate
        rseed : seed for randomization
    """

    def __init__(self, layer_dims, alpha=0.005, rseed=10):        
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims)
        param_dims = dict()
        for i in xrange(1, self.n_layers):
            if i < self.n_layers - 1:    
                param_dims['H_%d'%i] = (layer_dims[i], layer_dims[i])
            param_dims['W_%d'%i] = (layer_dims[i], layer_dims[i-1])
            param_dims['b_%d'%i] = (layer_dims[i],)
            
        param_dims_sparse = dict()
        NNBase.__init__(self, param_dims, param_dims_sparse, 
                        alpha=alpha, rseed=rseed)
                        
        self.H = [None] * (self.n_layers - 1)
        self.W = [None] * self.n_layers
        self.b = [None] * self.n_layers
        
        self.dH = [None] * (self.n_layers - 1)
        self.dW = [None] * self.n_layers
        self.db = [None] * self.n_layers
        for i in xrange(1, self.n_layers):
            if i < self.n_layers - 1:
                self.H[i] = self.params['H_%d'%i]
                self.dH[i] = self.grads['H_%d'%i]            
            self.W[i] = self.params['W_%d'%i]
            self.b[i] = self.params['b_%d'%i]
            self.dW[i] = self.grads['W_%d'%i]
            self.db[i] = self.grads['b_%d'%i]
            
        for i in xrange(1, self.n_layers):
            if i < self.n_layers-1:
                self.H[i][:] = random_weight_matrix(layer_dims[i], layer_dims[i])            
            self.W[i][:] = random_weight_matrix(layer_dims[i], layer_dims[i-1])
            

    def _acc_grads(self, xs, ys):
        T = len(xs)
        h = [[zeros(d)] for d in self.layer_dims[:-1]]
        y_hat = [None]
        
        for t in xrange(1, T+1):
            h[0].append(make_onehot(xs[t-1], self.layer_dims[0]))
            for l in xrange(1, self.n_layers-1):
                h[l].append(sigmoid(self.H[l].dot(h[l][t-1]) + self.W[l].dot(h[l-1][t]) + self.b[l]))
            y_hat.append(softmax(self.W[self.n_layers-1].dot(h[self.n_layers-2][t]) + self.b[self.n_layers-1]))
        
        delta = [zeros(d) for d in self.layer_dims]
        gamma = [None for d in self.layer_dims[:-1]]
        for t in xrange(T, 0, -1):
            delta[self.n_layers-1] = y_hat[t].copy()
            delta[self.n_layers-1][ys[t-1]] -= 1
            
            self.dW[self.n_layers-1] += outer(delta[self.n_layers-1], h[self.n_layers-2][t])
            self.db[self.n_layers-1] += delta[self.n_layers-1].copy()
            
            for l in xrange(self.n_layers-2, 0, -1):
                delta[l] = h[l][t] * (1 - h[l][t]) * self.W[l+1].T.dot(delta[l+1])
                if t == T:
                    gamma[l] = delta[l].copy()
                elif l == self.n_layers-2:
                    gamma[l] = delta[l] + h[l][t] * (1 - h[l][t]) * self.H[l].T.dot(gamma[l])
                else:
                    gamma[l] = delta[l] + h[l][t] * (1 - h[l][t]) * (self.H[l].T.dot(gamma[l]) + self.W[l+1].T.dot(gamma[l+1] - delta[l+1]))
                self.dH[l] += outer(gamma[l], h[l][t-1])
                self.dW[l] += outer(gamma[l], h[l-1][t])
                self.db[l] += gamma[l]

    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.
        """

        J = 0
        h = [zeros(d) for d in self.layer_dims[:-1]]
        for t in xrange(1, len(xs)+1):
            h[0] = make_onehot(xs[t-1], self.layer_dims[0])
            for l in xrange(1, self.n_layers - 1):
                h[l] = sigmoid(self.H[l].dot(h[l]) + self.W[l].dot(h[l-1]) + self.b[l])
            J += -log(softmax(self.W[self.n_layers-1].dot(h[self.n_layers-2]) + 
            self.b[self.n_layers-1])[ys[t-1]])
        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)


    def generate_sequence(self, init, end, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.
        Arguments:
            init = index of start word (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """

        J = 0 # total loss
        ys = [init]
        h = zeros(self.hdim)
        while ys[-1] != end:
            h = sigmoid(self.params.H.dot(h) + self.sparams.L[ys[-1]])
            p = softmax(self.params.U.dot(h))
            ys.append(multinomial_sample(p))
            J += -log(p[ys[-1]])
        return ys, J