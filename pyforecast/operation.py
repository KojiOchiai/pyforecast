# coding: utf-8

import numpy as np


def predict(ss, pure=True):
    return ss.predict(pure)


def update(ss, d, pure=True):
    return ss.update(d, pure)


def observe(ss, covariance=True):
    return ss.observe(covariance=covariance)


def row_vectorize(d):
    if len(d.shape) == 0:
        return np.array([[d]])
    elif len(d.shape) == 1:
        return d[:, None]
    elif len(d.shape) == 2:
        return d


def fillna(ss, d):
    d_ = d.copy()
    d_[np.isnan(d)] = observe(ss, covariance=False)[np.isnan(d)]
    return d_


class Filter():
    def __init__(self, system):
        self.system = system.copy()

    def __call__(self, d):
        self.system.predict()
        if np.isnan(d).all():
            return self.system
        else:
            d_ = fillna(self.system, row_vectorize(d))
            self.system.update(d_)
            return self.system


class Predictor():
    def __init__(self, system):
        self.system = system.copy()

    def __call__(self):
        self.system.predict()
        return self.system
