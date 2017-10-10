# coding: utf-8

import numpy as np
from .system import AbstractSystem


class NonLinear(AbstractSystem):
    '''NonLinear
    transition_function: function return distribution
    observe_function: function return distribution
    '''

    def __init__(self, transition_function, observe_function,
                 initial_state):
        self.transition_function = transition_function
        self.observe_function = observe_function
        self.state = initial_state
        self.n_sample = len(initial_state)

    def copy(self):
        return NonLinear(self.transition_function,
                         self.observe_function,
                         self.state)

    def state_update(self, next_state, pure=False):
        if pure:
            next_system = NonLinear(self.transition_function,
                                    self.observe_function,
                                    next_state)
            return next_system
        else:
            self.state = next_state
            return None

    def f_inv(self, w, u):
        if np.all(u < w):
            return 0
        return np.where(w < u)[0].max() + 1

    def resampling(self, w):
        N = self.n_sample
        u_list = np.random.uniform(0, 1 / N) + np.arange(N) / N
        w_cumsum = np.cumsum(w)
        w_cumsum /= w_cumsum[-1]
        return np.array([self.f_inv(w_cumsum, u) for u in u_list])

    def predict(self, pure=False):
        next_state = self.transition_function(self.state).sample(self.n_sample)
        return self.state_update(next_state, pure)

    def update(self, x, pure=False):
        ll = self.observe_function(self.state).likelihood(x, reduce='no')
        i = self.resampling(ll.sum(axis=0))
        next_state = self.state[i]
        return self.state_update(next_state, pure)

    def observe(self, n=1, covariance=True):
        if covariance:
            return self.observe_function(self.state)
        else:
            return self.observe_function(self.state).mu.mean(axis=1)[:, None]
