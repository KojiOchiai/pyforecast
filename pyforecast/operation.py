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


def update(ss, d):
    x = ss.state
    V = ss.covariance
    I = np.identity(V.shape[0])

    # K = V @ ss.H.T @ inv(ss.H @ V @ ss.H.T + ss.R)
    K = V @ solve((ss.H @ V @ ss.H.T + ss.R).T, (ss.H.T).T).T
    next_x = x + K @ (d - ss.H @ x)
    next_V = (I - K @ ss.H) @ V

    next_ss = ss.copy()
    next_ss.state = next_x
    next_ss.covariance = next_V
    return next_ss


def observe(ss, covariance=True):
    mean = ss.H @ ss.state
    if not covariance:
        return mean
    else:
        V = ss.covariance
        cov = ss.H @ V @ ss.H.T + ss.R
        return mean, cov


def fillna(ss, d):
    d_ = d.copy()
    d_[np.isnan(d)] = observe(ss, covariance=False)[np.isnan(d)]
    return d_


class Filter():
    def __init__(self, statespace):
        self.statespace = statespace.copy()

    def __call__(self, d):
        d_ = fillna(self.statespace, d)
        self.statespace = update(predict(self.statespace), d_)
        return self.statespace


class Predictor():
    def __init__(self, statespace):
        self.statespace = statespace.copy()

    def __call__(self):
        self.statespace = predict(self.statespace)
        return self.statespace
