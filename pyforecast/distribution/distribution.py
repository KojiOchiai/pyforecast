# coding: utf-8

import numpy as np


class Distribution():
    def __init__(self):
        super().__init__()

    def get_params(self):
        raise NotImplementedError()

    def likelihood(self, x, reduce='sum'):
        return np.exp(self.log_likelihood(x, reduce=reduce))

    def log_likelihood(self, x, reduce='sum'):
        raise NotImplementedError()

    def nll(self, x, reduce='sum'):
        return -self.log_likelihood(x, reduce=reduce)

    def kl(self, p):
        raise NotImplementedError()

    def check_reduce(self, reduce):
        if reduce not in ('sum', 'no'):
            raise ValueError(
                "only 'sum' and 'no' are valid for 'reduce', but '%s' is "
                'given' % reduce)
        else:
            return True

    def reduce(self, loss, reduce):
        if reduce == 'sum':
            return np.sum(loss[np.logical_not(np.isnan(loss))])
        else:
            return loss
