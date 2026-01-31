from typing import Tuple, Union

import numpy as np
from sortedcontainers import SortedDict

from ..tqwt import tqwt, itqwt, compute_wavelet_norms


class DualQDecomposition:
    """
    Resonance signal decomposition using two Q-factors for signal with noise.
    This is obtained by minimizing the cost function:

    || x - x1 - x2 ||_2^2 + lambda_1 ||w1||_1 + lambda_2 ||w2||_2

    References
    ----------
    .. [1] Selesnick, I. W. (2011). Resonance-based signal decomposition: A new sparsity-enabled signal analysis method.
           Signal Processing, 91(12), 2793-2809.

    .. [2] Selesnick, I. W. (2011). TQWT toolbox guide. Electrical and Computer Engineering, Polytechnic Institute of
           New York University. Available online at: http://eeweb.poly.edu/iselesni/TQWT/index.html


    Parameters
    ----------
    q1: float
    redundancy_1: float
    stages_1: int

    q2: float
    redundancy_2: float
    stages_2: int

    lambda_1: float
    lambda_2: float
    mu: float
    num_iterations: int

    compute_cost_function: bool

    """

    def __init__(
        self,
        q1: float,
        redundancy_1: float,
        stages_1: int,
        q2: float,
        redundancy_2: float,
        stages_2: int,
        lambda_1: float,
        lambda_2: float,
        mu: float,
        num_iterations: int,
        compute_cost_function: bool = False,
    ):
        # parameters of the first transform, define (i)tqwt lambdas
        self._q1 = q1
        self._redundancy_1 = redundancy_1
        self._stages_1 = stages_1
        self.tqwt1 = lambda x: tqwt(x, self._q1, self._redundancy_1, self._stages_1)
        self.itqwt1 = lambda w, n: itqwt(w, self._q1, self._redundancy_1, n)
        # parameters of the second transform, define (i)tqwt lambdas
        self._q2 = q2
        self._redundancy_2 = redundancy_2
        self._stages_2 = stages_2
        self.tqwt2 = lambda x: tqwt(x, self._q2, self._redundancy_2, self._stages_2)
        self.itqwt2 = lambda w, n: itqwt(w, self._q2, self._redundancy_2, n)
        # SALSA parameters
        self._lambda_1 = lambda_1
        self._lambda_2 = lambda_2
        self._mu = mu
        self._num_iterations = num_iterations

        self._history = None
        self._compute_cost_function = compute_cost_function

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a Dual-Q decomposition on the one-dimensional time-domain signal `x`.

        Parameters
        ----------
        x: np.ndarray, one-dimensional with even length
            Input signal

        Returns
        -------
        x1: np.ndarray with x.shape
            Signal component corresponding to the first transform
        x2: np.ndarray with x.shape
            Signal component corresponding to the second transform
        """
        assert len(x.shape) == 1

        n = x.shape[0]
        w1, w2 = self.tqwt1(x), self.tqwt2(x)
        d1, d2 = (
            np.array([np.zeros(s.shape, s.dtype) for s in w], dtype=object)
            for w in [w1, w2]
        )
        u1, u2 = (
            np.array([np.zeros(s.shape, s.dtype) for s in w], dtype=object)
            for w in [w1, w2]
        )
        t1 = (
            self._lambda_1
            * compute_wavelet_norms(n, self._q1, self._redundancy_1, self._stages_1)
            / (2 * self._mu)
        )
        t2 = (
            self._lambda_2
            * compute_wavelet_norms(n, self._q2, self._redundancy_2, self._stages_2)
            / (2 * self._mu)
        )

        for iter_idx in range(self._num_iterations):
            for j in range(self._stages_1 + 1):
                u1[j] = self.soft_threshold(w1[j] + d1[j], t1[j]) - d1[j]
            for j in range(self._stages_2 + 1):
                u2[j] = self.soft_threshold(w2[j] + d2[j], t2[j]) - d2[j]

            c = (x - self.itqwt1(u1, n) - self.itqwt2(u2, n)) / (self._mu + 2)
            d1, d2 = self.tqwt1(c), self.tqwt2(c)

            for j in range(self._stages_1 + 1):
                w1[j] = d1[j] + u1[j]
            for j in range(self._stages_2 + 1):
                w2[j] = d2[j] + u2[j]

            if self._compute_cost_function:
                self.update_history(iter_idx, w1, w2, t1, t2, x)

        return self.itqwt1(w1, n), self.itqwt2(w2, n)

    @staticmethod
    def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
        y = np.abs(x) - thresh
        y[np.where(y < 0)] = 0
        return y / (y + thresh) * x

    def update_history(self, iter_idx, w1, w2, t1, t2, x) -> None:
        """Compute the cost function and store its value in the history dict."""
        if iter_idx == 0:  # re-initialize the history for every run
            self._history = SortedDict()

        residual = x - self.itqwt1(w1, x.shape[0]) - self.itqwt2(w2, x.shape[0])
        cost_function = np.sum(np.abs(residual) ** 2)
        for j in range(self._stages_1 + 1):
            cost_function += t1[j] * np.sum(np.abs(w1[j]))
        for j in range(self._stages_2 + 1):
            cost_function += t2[j] * np.sum(np.abs(w2[j]))
        self._history[iter_idx] = cost_function

    @property
    def history(self) -> Union[SortedDict, None]:
        """
        If `compute_cost_function` is True a SortedDict mapping the iteration steps to the value of the cost function is
        returned. Returns None if `compute_cost_function` is False.
        """
        return self._history
