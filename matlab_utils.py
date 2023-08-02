import random
import numpy as np
from itertools import accumulate

# Matlab functions translated to python:

def cell(rows, cols):
    """ Pre-allocate a 2D matrix of empty lists. """
    return [ [ [] for i in range(cols) ] for j in range(rows) ]


def randi(imax, size):
    if isinstance(size, int):
        if isinstance(imax, int):
            return random.choices(range(imax), k=size)
        elif isinstance(imax, tuple):
            return random.choices(range(imax[0], imax[1]), k=size)
    elif isinstance(size, tuple) or isinstance(size, list):
        mat = np.random.randint(low=1, high=imax+1, size=size)
        return mat
    return None


def randperm(n, **kwargs):
    k = kwargs.get('k_', n)
    return list(np.random.choice(range(n), size=k, replace=False))


def cumsum(list_):
    return list(accumulate(list_))


def setdiff(A, B):
    return [item for item in A if not item in B]


def randsample(n,k,**kwargs):

    replacement = kwargs.get('repl', False)
    if k > n:
        replacement = True

    weights = kwargs.get('weights', None)

    # https://www.mathworks.com/help/stats/randsample.html#d124e839496
    # uses a vector of non-negative weights, w, whose length is n, 
    # to determine the probability that an integer i is selected as an entry for y.

    if weights:
        return random.choices(range(n), weights=weights, k=k)
    else:
        if replacement:
            return random.choices(range(n), k=k)
        else:
            return random.sample(range(n), n)


def rand(size_1, size_2):
    rand_ = np.random.uniform(low=0, high=1, size=(size_1, size_2))
    if size_2 == 1:
        rand_ = list(rand_.T[0])
    return rand_


def sort(array, dimension):

    if (dimension == 1) or (dimension == 2):

        if dimension == 1:
            array = array.T

        array_sorted = []
        for k in range(array.shape[0]):
            lst_append = list(array[k])
            lst_append.sort()
            array_sorted.append(lst_append)
        array_sorted = np.array(array_sorted)

        if dimension == 1:
            array_sorted = array_sorted.T

        return array_sorted

    else:
        return array


def histc(x, bins):
    if not isinstance(x, list):
        x = [x]
    x_idx = [0]*len(x)
    for idx, xx in enumerate(x):
        found = False
        for i in range(len(bins) - 1):
            if (xx >= bins[i]) and (xx < bins[i+1]) and not found:
                found = True
                x_idx[idx] = i + 1
    return x_idx
