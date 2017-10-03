# coding: utf-8

import numpy as np


def predict(ss):
    return ss.predict()


def update(ss, d):
    return ss.update(d)


def observe(ss, covariance=True):
    return ss.observe(covariance)


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
    def __init__(self, statespace):
        self.statespace = statespace.copy()

    def __call__(self, d):
        if np.isnan(d).all():
            self.statespace = predict(self.statespace)
            return self.statespace
        else:
            d_ = fillna(self.statespace, row_vectorize(d))
            self.statespace = update(predict(self.statespace), d_)
            return self.statespace


class Predictor():
    def __init__(self, statespace):
        self.statespace = statespace.copy()

    def __call__(self):
        self.statespace = predict(self.statespace)
        return self.statespace
