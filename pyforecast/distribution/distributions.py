# coding: utf-8

import math
import numpy as np
from scipy import stats
from .distribution import Distribution


class Gaussian(Distribution):
    def __init__(self, mu, var, n_sample=1, clip=False,
                 clip_min=0.01, clip_max=10):
        super().__init__()
        if clip:
            var = np.clip(var, clip_min, clip_max)
        self.mu = mu
        self.var = var
        self.ln_var = np.log(var)
        self.n_sample = n_sample

    def get_params(self):
        return (self.mu, self.ln_var)

    def log_likelihood(self, x, reduce='sum'):
        self.check_reduce(reduce)
        x_prec = np.exp(-self.ln_var)
        x_diff = x - self.mu
        x_power = (x_diff * x_diff) * x_prec * -0.5
        loss = - (self.ln_var + math.log(2 * math.pi)) / 2 + x_power
        return self.reduce(loss, reduce)

    def sample(self, n_sample=None):
        N = n_sample or self.n_sample
        sd = np.sqrt(self.var)
        return np.random.normal(np.tile(self.mu, (N, 1)),
                                np.tile(sd, (N, 1)))


class Poisson(Distribution):
    def __init__(self, mu, n_sample=1):
        super().__init__()
        if np.any(mu < 0):
            raise ValueError('mu must be grater than or equal to 0')
        self.mu = mu
        self.n_sample = n_sample

    def get_params(self):
        return self.mu

    def log_likelihood(self, x, reduce='sum'):
        self.check_reduce(reduce)
        loss = stats.poisson.logpmf(x, self.mu)
        return self.reduce(loss, reduce)

    def sample(self, n_sample=None):
        N = n_sample or self.n_sample
        return np.random.poisson(np.tile(self.mu, (N, 1)))


class Uniform(Distribution):
    def __init__(self, xmin, xmax, n_sample=1):
        super().__init__()
        self.min = xmin
        self.max = xmax
        self.n_sample = n_sample

    def get_params(self):
        return (self.min, self.max)

    def log_likelihood(self, x, reduce='sum'):
        self.check_reduce(reduce)
        ll = np.log(1 / (self.max - self.min))
        in_range = np.logical_and(self.min <= x, x <= self.max)
        loss = in_range * ll
        loss[loss == 0] = np.NaN
        return self.reduce(loss, reduce)

    def sample(self, n_sample=None):
        N = n_sample or self.n_sample
        return np.random.uniform(np.tile(self.min, (N, 1)),
                                 np.tile(self.max, (N, 1)))


def gaussian(mu, var):
    d = Gaussian(mu, var)

    def dist(n=1):
        return d.sample(n).T[0]

    return dist


def poisson(mu):
    d = Poisson(mu)

    def dist(n=1):
        return d.sample(n).T[0]

    return dist


def uniform(xmin, xmax):
    d = Uniform(xmin, xmax)

    def dist(n=1):
        return d.sample(n).T[0]

    return dist
