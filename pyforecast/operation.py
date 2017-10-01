# coding: utf-8

import numpy as np
from numpy.linalg import solve


def predict(ss):
    x = ss.state
    V = ss.covariance

    next_x = ss.F @ x
    next_V = ss.F @ V @ ss.F.T + ss.G @ ss.Q @ ss.G.T

    next_ss = ss.copy()
    next_ss.state = next_x
    next_ss.covariance = next_V
    return next_ss


def postdict(ss, xf, Vf):
    x = ss.state
    V = ss.covariance

    A = V @ solve(V.T, ss.F.T).T
    prev_x = x + A @ (xf - x)
    prev_V = V + A @ (Vf - V) @ A.T

    prev_ss = ss.copy()
    prev_ss.state = prev_x
    prev_ss.covariance = prev_V
    return prev_ss


def update(ss, d):
    x = ss.state
    y = d - ss.offset
    V = ss.covariance
    I = np.identity(V.shape[0])

    # K = V @ ss.H.T @ inv(ss.H @ V @ ss.H.T + ss.R)
    K = V @ solve((ss.H @ V @ ss.H.T + ss.R).T, (ss.H.T).T).T
    next_x = x + K @ (y - ss.H @ x)
    next_V = (I - K @ ss.H) @ V

    next_ss = ss.copy()
    next_ss.state = next_x
    next_ss.covariance = next_V
    return next_ss


def observe(ss, covariance=True):
    mean = ss.H @ ss.state + ss.offset
    if not covariance:
        return mean
    else:
        V = ss.covariance
        cov = ss.H @ V @ ss.H.T + ss.R
        return mean, cov


def row_vectorize(d):
    if len(d.shape) == 2:
        return d
    elif len(d.shape) == 1:
        return d[:, None]


def fillna(ss, d):
    d_ = d.copy()
    d_[np.isnan(d)] = observe(ss, covariance=False)[np.isnan(d)]
    return d_


class Filter():
    def __init__(self, statespace):
        self.statespace = statespace.copy()

    def __call__(self, d):
        d_ = fillna(self.statespace, row_vectorize(d))
        self.statespace = update(predict(self.statespace), d_)
        return self.statespace


class Predictor():
    def __init__(self, statespace):
        self.statespace = statespace.copy()

    def __call__(self):
        self.statespace = predict(self.statespace)
        return self.statespace
