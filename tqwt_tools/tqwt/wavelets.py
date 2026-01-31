from copy import deepcopy
from typing import Callable

import numpy as np

from .itqwt import itqwt
from .tqwt import tqwt


def compute_wavelets(n: int, q: float, redundancy: float, stages: int) -> np.ndarray:
    n_zeros = np.zeros(n)
    wavelet_shaped_zeros = tqwt(n_zeros, q, redundancy, stages)
    wavelets = np.array([None for _ in range(stages + 1)], dtype=object)

    for j_i in range(stages + 1):
        w = deepcopy(wavelet_shaped_zeros)
        m = int(round(w[j_i].shape[0] / 2))
        w[j_i][m] = 1.0
        wavelet = itqwt(w, q, redundancy, n)
        wavelets[j_i] = wavelet

    return wavelets


def compute_wavelet_norms(
    n: int,
    q: float,
    redundancy: float,
    stages: int,
    norm_function: Callable = np.linalg.norm,
) -> np.ndarray:
    return np.array(
        [norm_function(w_j) for w_j in compute_wavelets(n, q, redundancy, stages)]
    )
