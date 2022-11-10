"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from joblib import wrap_non_picklable_objects

import talib as ta
import pandas as pd
from scipy import stats
from copy import copy

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity, is_ts=False, params_need : list =None):
        self.function = function
        self.name = name
        self.arity = arity

        self.is_ts = is_ts
        self.d = 0
        self.params_need = params_need

    def __call__(self, *args):
        if not self.is_ts:
            return self.function(*args)
        else:
            if self.d == 0:
                raise AttributeError('Please set "d"')
            else:
                return self.function(*args, self.d)
    
    def set_d(self, d):
        self.d = d
        self.name += '_%d' % self.d

def make_ts_function(function, d_ls, random_state):
    """
    Parameters
    ----------
    function: _Function

    d_ls: list
        参数 'd' 可选范围.

    random_state: RandomState instance
        随机数生成器.

    """
    d = random_state.randint(len(d_ls))
    d = d_ls[d]
    function_ = copy(function)
    function_.set_d(d)
    return function_


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

def _ts_delay(x1, d):
    return pd.Series(x1).shift(d).values
ts_delay1 = _Function(function=_ts_delay, name='ts_delay', arity=1, is_ts=True)

def _ts_delta(x1, d):
    return x1 - _ts_delay(x1, d)
ts_delta1 = _Function(function=_ts_delta, name='ts_delta', arity=1, is_ts=True)

def _ts_min(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).min()
ts_min1 = _Function(function=_ts_min, name='ts_min', arity=1, is_ts=True)

def _ts_max(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).max()
ts_max1 = _Function(function=_ts_max, name='ts_max', arity=1, is_ts=True)

def _ts_argmin(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).apply(lambda x: x.argmin())
ts_argmin1 = _Function(function=_ts_argmin, name='ts_argmin', arity=1, is_ts=True)

def _ts_argmax(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).apply(lambda x: x.argmax())
ts_argmax1 = _Function(function=_ts_argmax, name='ts_argmax', arity=1, is_ts=True)

def _ts_rank(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).apply(
        lambda x: stats.percentileofscore(x, x[x.last_valid_index()]) / 100
    )
ts_rank1 = _Function(function=_ts_rank, name='ts_rank', arity=1, is_ts=True)

def _ts_sum(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).sum()
ts_sum1 = _Function(function=_ts_sum, name='ts_sum', arity=1, is_ts=True)

def _ts_stddev(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).std()
ts_stddev1 = _Function(function=_ts_stddev, name='ts_stddev', arity=1, is_ts=True)

def _ts_corr(x1, x2, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).corr(pd.Series(x2))
ts_corr2 = _Function(function=_ts_corr, name='ts_corr', arity=2, is_ts=True)

def _ts_cov(x1, x2, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).cov(pd.Series(x2))
ts_cov2 = _Function(function=_ts_cov, name='ts_cov', arity=2, is_ts=True)

def _ts_mean_return(x1, d):
    return pd.Series(x1).pct_change().rolling(d, min_periods=int(d / 2)).mean()
ts_mean_return1 = _Function(function=_ts_mean_return, name='ts_mean_return',
                            arity=1, is_ts=True)


def _ts_dema1(x1, d):
    try:
        return ta.DEMA(x1, d)
    except Exception:
        return np.zeros_like(x1)

def _ts_kama1(x1, d):
    try:
        return ta.KAMA(x1, d)
    except Exception:
        return np.zeros_like(x1)

def _ts_ma1(x1, d):
    try:
        return ta.MA(x1, d)
    except Exception:
        return np.zeros_like(x1)

def _ts_midpoint1(x1, d):
    try:
        return ta.MIDPOINT(x1, d)
    except Exception:
        return np.zeros_like(x1)

def _ts_beta2(x1, x2, d):
    try:
        return ta.BETA(x1, x2, d)
    except Exception:
        return np.zeros_like(x1)

def _ts_lr_angle1(x1, d):
    try:
        return ta.LINEARREG_ANGLE(x1, d)
    except Exception:
        return np.zeros_like(x1)

def _ts_lr_intercept1(x1, d):
    try:
        return ta.LINEARREG_INTERCEPT(x1, d)
    except Exception:
        return np.zeros_like(x1)

def _ts_ts_lr_slope1(x1, d):
    try:
        return ta.LINEARREG_SLOPE(x1, d)
    except Exception:
        return np.zeros_like(x1)

def _ht_dcperiod(x1):
    try:
        return ta.HT_DCPERIOD(x1)
    except Exception:
        return np.zeros_like(x1)

ht_dcperiod1 = _Function(function=_ht_dcperiod, name='HT_DCPERIOD', arity=1)

ts_dema1 = _Function(function=_ts_dema1, name='DEMA', arity=1, is_ts=True)
ts_kama1 = _Function(function=_ts_kama1, name='KAMA', arity=1, is_ts=True)
ts_ma1 = _Function(function=_ts_ma1, name='MA', arity=1, is_ts=True)
ts_midpoint1 = _Function(function=_ts_midpoint1, name='MIDPOINT', arity=1, is_ts=True)
ts_beta2 = _Function(function=_ts_beta2, name='BETA', arity=2, is_ts=True)
ts_lr_angle1 = _Function(function=_ts_lr_angle1, name='LR_ANGLE',
                         arity=1, is_ts=True)
ts_lr_intercept1: _Function = _Function(function=_ts_lr_intercept1,
                                        name='LR_INTERCEPT', arity=1, is_ts=True)
ts_lr_slope1 = _Function(function=_ts_ts_lr_slope1, name='LR_SLOPE',
                         arity=1, is_ts=True)


fixed_midprice = _Function(function=ta.MIDPRICE, name='midprice', arity=0, is_ts=True,
                           params_need=['high', 'low'])
fixed_aroonosc = _Function(function=ta.AROONOSC, name='AROONOSC', arity=0, is_ts=True,
                           params_need=['high', 'low'])
fixed_willr = _Function(function=ta.WILLR, name='WILLR', arity=0, is_ts=True,
                        params_need=['high', 'low', 'close'])
fixed_cci = _Function(function=ta.CCI, name='CCI', arity=0, is_ts=True,
                      params_need=['high', 'low', 'close'])
fixed_adx = _Function(function=ta.ADX, name='ADX', arity=0, is_ts=True,
                      params_need=['high', 'low', 'close'])
fixed_mfi = _Function(function=ta.MFI, name='MFI', arity=0, is_ts=True,
                      params_need=['high', 'low', 'close', 'volume'])
fixed_natr = _Function(function=ta.NATR, name='NATR', arity=0, is_ts=True,
                       params_need=['high', 'low', 'close'])

add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

_ts_function_map = {
    'ts_delay': ts_delay1,
    'ts_delta': ts_delta1,
    'ts_min': ts_min1,
    'ts_max': ts_max1,
    'ts_argmin': ts_argmin1,
    'ts_argmax': ts_argmax1,
    'ts_rank': ts_rank1,
    'ts_stddev': ts_stddev1,
    'ts_corr': ts_corr2,
    'ts_cov': ts_cov2,
    'ts_mean_return': ts_mean_return1,

    'DEMA': ts_dema1,
    'KAMA': ts_kama1,
    'MA': ts_ma1,
    'MIDPOINT': ts_midpoint1,
    'BETA': ts_beta2,
    'LR_ANGLE': ts_lr_angle1,
    'LR_INTERCEPT': ts_lr_intercept1,
    'LR_SLOPE': ts_lr_slope1,
}

_fixed_function_map = {
    'MIDPRICE': fixed_midprice,
    'AROONOSC': fixed_aroonosc,
    'WILLR': fixed_willr,
    'CCI': fixed_cci,
    'ADX': fixed_adx,
    'MFI': fixed_mfi,
    'NATR': fixed_natr
}

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'HT_DCPERIOD': ht_dcperiod1
}
