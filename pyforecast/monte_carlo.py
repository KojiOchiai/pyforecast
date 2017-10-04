# coding: utf-8

import numpy as np
import pyforecast as pf


def fit(statespace, data, test=1, sample=1, **params):
    ss = statespace.copy()
    if len(params) == 0:
        return params
    tune_param_names = params.keys()
    tune_params = []
    base_args = params2args(ss.get_params())
    base_args.update(params)

    for i in range(sample):
        param_samples = ss.sample_params(**base_args)
        ss_sample = pf.StateSpace(**params2args(param_samples))
        ll = log_likelihood(ss_sample, data, test=test)
        tmp_params = {k: param_samples[k] for k in tune_param_names}
        tmp_params['ll'] = ll
        tune_params.append(tmp_params)

    return tune_params[np.argmax([t['ll'] for t in tune_params])]


def log_likelihood(ss, data, test=1, reduce='sum'):
    study = data[:-test]
    test = data[-test:]

    filter_ = pf.Filter(ss)
    [filter_(d) for d in study]
    predictor = pf.Predictor(filter_.statespace)
    predicted = [pf.observe(predictor()) for i in range(test.shape[0])]

    predicted_mean = np.array([p[0] for p in predicted])[:, :, 0]
    predicted_cov = np.array([p[1] for p in predicted])
    predicted_var = np.array([np.diag(c) for c in predicted_cov])

    mg = pf.Gaussian(predicted_mean, predicted_var)
    return mg.log_likelihood(test, reduce=reduce)


def params2args(ss_params):
    ss_params.pop('n_dim_state')
    ss_params.pop('n_dim_observe')
    ss_params['initial_state'] = ss_params.pop('state')
    ss_params['initial_covariance'] = ss_params.pop('covariance')
    return ss_params
