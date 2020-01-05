from typing import Tuple

import numpy as np
from scipy.fftpack import fft, ifft

from tqwt_tools.tqwt.check_params import check_params


def tqwt(x: np.ndarray, q: float, redundancy: float, stages: int) -> np.ndarray:
    """
    Tunable-Q Wavelet transform TODO: more docs for tqwt

    Parameters
    ----------
    x: array like, shape (n,)
        Input signal, needs to be of even length
    q: float
        Q-Factor. The Q-factor, denoted Q, affects the oscillatory behavior the wavelet; specifically, Q affects the
        extent to which the oscillations of the wavelet are sustained. Roughly, Q is a measure of the number of
        oscillations the wavelet exhibits. For Q, a value of 1.0 or greater can be specified. The definition of the
        Q-factor of an oscillatory pulse is the ratio of its center frequency to its bandwidth.
    redundancy: float
        Oversampling rate (redundancy). Redundancy of the TQWT when it is computed using
        infinitely many levels. Here `redundancy means total over-sampling rate of the transform (the total number of
        wavelet coefficients divided by the length of the signal to which the TQWT is applied.) The specified value of
        must be greater than 1.0, and a value of 3.0 or greater is recommended. (When it is close to 1.0, the wavelet
        will not be well localized in time — it will have excessive ringing which is generally considered undesirable.)
        The actual redundancy will be somewhat different than the parameter because the transform can actually be
        computed using only a finite number of levels.
    stages: int
        The number of stages (or levels) of the wavelet transform is denoted by stages. The transform consists of a
        sequence of two-channel filter banks, with the low-pass output of each filter bank being used as the input to
        the successive filter bank. The parameter `stages` denotes the number of filter banks. Each output signal
        constitutes one subband of the wavelet transform. There will be J + 1 subbands: the high-pass filter output
        signal of each filter bank, and the low-pass filter output signal of the final filter bank.

    Returns
    -------
    w: np.ndarray with dtype np.object
        Wavelet coefficients.

    Examples
    --------
    >>> # verify perfect reconstruction
    >>> import numpy as np
    >>> from tqwt_tools import tqwt
    >>> q = 4; redundancy = 3; stages = 3      # parameters
    >>> n = 200                  # signal length
    >>> x = np.random.randn(n)    # test signal (white noise)
    >>> w = tqwt(x, q, redundancy, stages)       # wavelet transform
    >>> y = itqwt(w, q, redundancy, N);      # inverse wavelet transform
    >>> max(abs(x - y))          # reconstruction error
    """
    check_params(q, redundancy, stages)
    if x.shape[0] % 2 or len(x.shape) != 1:
        raise ValueError("Input signal x needs to be one dimensional and of even length!")
    x = np.asarray(x)

    beta = float(2 / (q + 1))
    alpha = float(1 - beta / redundancy)
    n = x.shape[0]

    max_num_stages = int(np.floor(np.log(beta * n / 8) / np.log(1 / alpha)))

    if stages > max_num_stages:
        if max_num_stages > 0:
            raise ValueError("Too many subbands, reduce subbands to " + str(max_num_stages))
        else:
            raise ValueError("Too many subbands specified, increase signal length")

    fft_of_x = fft(x) / np.sqrt(n)  # unitary DFT

    w = []                          # init list of wavelet coefficients

    for subband_idx in range(1, stages + 1):
        n0 = 2 * round(alpha ** subband_idx * n / 2)
        n1 = 2 * round(beta * alpha ** (subband_idx - 1) * n / 2)
        fft_of_x, w_subband = analysis_filter_bank(fft_of_x, n0, n1)
        w.append(ifft(w_subband) * np.sqrt(len(w_subband)))  # inverse unitary DFT

    w.append(ifft(fft_of_x) * np.sqrt(len(fft_of_x)))      # inverse unitary DFT
    return np.array(w, dtype=np.object)


def analysis_filter_bank(x: np.ndarray, n0: int, n1: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-channel analysis filter bank operating on a frequency domain input x. This function is used
    iteratively by `tqwt`.

    Parameters
    ----------
    x: np.ndarray
        Input vector (frequency domain)
    n0: int
        length of the lp_subband
    n1: int
        length of the hp_subband

    Returns
    -------
    lp_subband: np.ndarray
        low-pass output of the filter bank in the frequency domain
    hp_subband: np.ndarray
        high-pass output of the filter bank in the frequency domain
    """
    x = np.array(x)
    n = x.shape[0]                  # len(x)

    p = int((n-n1) / 2)             # pass-band
    t = int((n0 + n1 - n) / 2 - 1)  # transition-band
    s = int((n - n0) / 2)           # stop-band

    # transition band function
    v = np.arange(start=1, stop=t+1) / (t+1) * np.pi
    transit_band = (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2.0

    # low-pass subband
    lp_subband = np.zeros(n0, dtype=x.dtype)
    lp_subband[0] = x[0]                                                  # DC-term
    lp_subband[1:p+1] = x[1:p + 1]                                        # pass-band
    lp_subband[1+p:p+t+1] = x[1 + p:p + t + 1] * transit_band             # transition-band
    lp_subband[int(n0 / 2)] = 0                                           # nyquist
    lp_subband[n0-p-t:n0-p] = x[n - p - t:n - p] * np.flip(transit_band)  # transition-band (negative frequencies)
    lp_subband[n0-p:] = x[n - p:]                                         # pass-band (negative frequencies)

    # high-pass subband
    hp_subband = np.zeros(n1, dtype=x.dtype)
    hp_subband[0] = 0                                                     # DC-term
    hp_subband[1:t+1] = x[1 + p:t + p + 1] * np.flip(transit_band)        # transition-band
    hp_subband[t+1:s+1+t] = x[p + t + 1:p + t + s + 1]                    # pass-band
    if n // 2 == 0:                                                       # nyquist if N is even
        hp_subband[n1/2] = x[n / 2]
    hp_subband[n1-t-s-1:n1-t] = x[n - p - t - s - 1:n - p - t]            # pass-band (negative frequencies)
    hp_subband[n1-t:n1] = x[n - p - t:n - p] * transit_band               # transition-band (negative frequencies)

    return lp_subband, hp_subband
