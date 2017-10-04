# coding: utf-8

import math
import numpy as np
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

    def kl(self, p):
        """Calculate KL-divergence between given two gaussian.
        D_{KL}(P||Q)=\frac{1}{2}\Bigl[(\mu_1-\mu_2)^T
        \Sigma_2^{-1}(\mu_1-\mu_2)
        + tr\bigl\{\Sigma_2^{-1}\Sigma_1 \bigr\}
        + \log\frac{|\Sigma_2|}{|\Sigma_1|} - d \Bigr]
        """
        assert isinstance(p, Gaussian)
        mu1, ln_var1 = self.mu, self.ln_var
        mu2, ln_var2 = p.mu, p.ln_var
        assert mu1.data.shape == mu2.data.shape
        assert ln_var1.data.shape == ln_var2.data.shape

        d = mu1.size  # N x D
        var1 = np.exp(ln_var1)
        var2 = np.exp(ln_var2)
        return (np.sum((mu1 - mu2) * (mu1 - mu2) / var2)
                + np.sum(var1 / var2)
                + np.sum(ln_var2)
                - np.sum(ln_var1) - d) * 0.5

    def entropy(self):
        batch, dim = self.ln_var.shape
        return (np.sum(self.ln_var)
                + np.log(2 * np.pi * np.e) * 0.5 * dim * batch)

    def sample(self, n_sample=None):
        N = n_sample or self.n_sample
        sd = np.sqrt(self.var)
        return np.random.normal(np.tile(self.mu, (N, 1)),
                                np.tile(sd, (N, 1)))


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


def uniform(xmin, xmax):
    d = Uniform(xmin, xmax)

    def dist(n=1):
        return d.sample(n).T[0]

    return dist
