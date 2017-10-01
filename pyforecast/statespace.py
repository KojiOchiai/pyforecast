# coding: utf-8

import types
import numpy as np
from scipy.misc import comb


class StateSpace():
    # state space
    #
    # F: transition matrix
    # G: noise transition matrix
    # H: observation matrix
    # Q: transition covariance
    # R: observation covariance
    def __init__(self, F, G, H, Q=np.ones, R=np.ones, offset=np.zeros,
                 initial_state=np.zeros, initial_covariance=np.ones):
        n_dim_state = F.shape[0]
        n_dim_observe = H.shape[0]

        def isfunction(x):
            return (isinstance(x, types.FunctionType) or
                    isinstance(x, types.BuiltinFunctionType))

        if isfunction(Q):
            Q = np.diag(Q(G.shape[1]))

        if isfunction(R):
            R = np.diag(R(n_dim_observe))

        if isfunction(offset):
            offset = offset(n_dim_observe)[:, None]

        if isfunction(initial_state):
            initial_state = initial_state(n_dim_state)[:, None]

        if isfunction(initial_covariance):
            initial_covariance = np.diag(initial_covariance(n_dim_state))

        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        self.state = initial_state
        self.covariance = initial_covariance
        self.offset = offset
        self.n_dim_state = n_dim_state
        self.n_dim_observe = n_dim_observe

    def compose(self, ss):
        F = compose_matrix(self.F, ss.F)
        G = compose_matrix(self.G, ss.G)
        H = np.concatenate([self.H, ss.H], axis=1)
        Q = compose_matrix(self.Q, ss.Q)
        R = (self.R + ss.R) / 2
        offset = np.max(np.concatenate([self.offset, ss.offset], axis=1),
                        axis=1)[:, None]
        state = np.concatenate([self.state, ss.state])
        return StateSpace(F, G, H, Q, R, offset, state)

    def __add__(self, other):
        return self.compose(other)

    def copy(self):
        F = self.F.copy()
        G = self.G.copy()
        H = self.H.copy()
        Q = self.Q.copy()
        R = self.R.copy()
        offset = self.offset.copy()
        state = self.state.copy()
        return StateSpace(F, G, H, Q, R, offset, state)


class Trend(StateSpace):
    def __init__(self, k, dim=1):
        c = np.array([(-1) ** ((i + 1) % 2) * comb(k, i)
                      for i in range(1, k + 1)])[None, :]
        F = np.concatenate([c, time_shift(k)])
        G = np.zeros([k, 1])
        G[0, 0] = 1
        H = np.zeros([1, k])
        H[0, 0] = 1
        super().__init__(F, G, H)


class Period(StateSpace):
    def __init__(self, period):
        c = np.array([-1 for i in range(1, period + 1)])[None, :]
        F = np.concatenate([c, time_shift(period)])
        G = np.zeros([period, 1])
        G[0, 0] = 1
        H = np.zeros([1, period])
        H[0, 0] = 1
        super().__init__(F, G, H)


def compose_matrix(A, B):
    rA, cA = A.shape
    rB, cB = B.shape
    Z1 = np.zeros([rA, cB])
    Z2 = np.zeros([rB, cA])
    return np.concatenate([np.concatenate([A, Z1], axis=1),
                           np.concatenate([Z2, B], axis=1)])


def time_shift(dim, shift=1):
    return np.concatenate([np.identity(dim - shift),
                           np.zeros([dim - shift, shift])],
                          axis=1)


def expand_matrix_dim(M, dim=1):
    r, c = M.shape
    expanded = np.zeros([dim * r, dim * c])
    for d in range(dim):
        expanded[d::dim, d::dim] = M
    return expanded


def expand_vector_dim(M, dim=1):
    r, c = M.shape
    expanded = np.zeros([dim * r, c])
    for d in range(dim):
        expanded[d::dim, :] = M
    return expanded


def expand_ss_dim(ss, dim=1):
    F = expand_matrix_dim(ss.F, dim)
    G = expand_matrix_dim(ss.G, dim)
    H = expand_matrix_dim(ss.H, dim)
    Q = expand_matrix_dim(ss.Q, dim)
    R = expand_matrix_dim(ss.R, dim)
    state = expand_vector_dim(ss.state, dim)
    offset = expand_vector_dim(ss.offset, dim)
    return StateSpace(F, G, H, Q, R, offset, state)
