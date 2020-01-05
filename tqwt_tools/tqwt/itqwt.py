import numpy as np
from scipy.fftpack import fft, ifft


def itqwt(w: np.ndarray, q: float, redundancy: float, n: int) -> np.ndarray:
    """
    Inverse Tunable-Q Wavelet Transform

    Parameters
    ----------
    w: np.ndarray with dtype np.object
        Wavelet coefficients for inverse transform
    q: float
        Q-Factor of the `tqwt` used for the forward transform. Greater or equal than 1.
    redundancy: float
        Parameter determining overlap ov the bands, s. `tqwt` docs for more info. Greater or equal than 1.
    n: int
        length of the original time-domain signal in samples.

    Returns
    -------
    y: np.ndarray:
        Time-domain signal

    """
    w = np.array(w, dtype=np.object)  # behaving sort of like a cell array in matlab

    # scaling factors
    beta = 2.0 / (q + 1)
    alpha = 1.0 - beta / redundancy
    num_subbands = w.shape[0]

    y = fft(w[num_subbands-1]) / np.sqrt(w[num_subbands-1].shape[0])                  # unitary DFT

    for subband_idx in reversed(range(num_subbands-1)):
        W = fft(w[subband_idx]) / np.sqrt(len(w[subband_idx]))  # unitary DFT
        m = int(2 * round(alpha ** subband_idx * n/2))
        y = synthesis_filter_bank(y, W, m)

    return np.real_if_close(ifft(y) * np.sqrt(y.shape[0]))      # inverse unitary DFT, discard small imaginary part


def synthesis_filter_bank(lp_subband: np.ndarray, hp_subband: np.ndarray, n: int) -> np.ndarray:
    """
    Complementary function for the `analysis_filter_bank`. Used iteratively by the `itqwt`
    
    Parameters
    ----------
    lp_subband: np.ndarray
        Low-pass subband (frequency-domain)
    hp_subband: np.ndarray
        High-pass subband (frequency-domain)
    n: int
        Length of the output in samples (frequency-domain)
    """
    n0 = lp_subband.shape[0]
    n1 = hp_subband.shape[0]

    p = int((n - n1) / 2)            # pass-band
    t = int((n0 + n1 - n) / 2 - 1)   # transition band
    s = int((n - n0) / 2)            # stop-band

    # transition band function
    v = np.arange(start=1, stop=t + 1) / (t + 1) * np.pi
    trans = (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2

    # low-pass subband
    y0 = np.zeros(n, dtype=np.complex)
    y0[0] = lp_subband[0]                                       # DC-term
    y0[1:p+1] = lp_subband[1:p + 1]                               # passband
    y0[1+p:p+t+1] = lp_subband[1 + p:p + t + 1] * trans               # transition-band
    y0[p+t+1:p+t+s+1] = np.zeros((p+t+s+1) - (p+t+1))         # stop-band
    if n // 2 == 0:
        y0[n/2] = 0                                           # Nyquist if even length
    y0[n-p-t-s:n-p-t] = np.zeros((n-p-t) - (n-p-t-s))         # stop-band (negative frequency)
    y0[n-p-t:n-p] = lp_subband[n0 - p - t:n0 - p] * np.flip(trans)  # transition band (negative frequencies)
    y0[n-p:] = lp_subband[n0 - p:]                                # passband (negative frequency)

    # high-pass subband
    y1 = np.zeros(n, dtype=np.complex)
    y1[0] = 0                                                 # DC-term
    y1[1:p+1] = np.zeros(p)                                   # stop-band
    y1[1+p:t+p+1] = hp_subband[1:t + 1] * np.flip(trans)          # transition-band
    y1[p+t+1:p+t+s+1] = hp_subband[t + 1:s + 1 + t]                   # passband
    if n // 2 == 0:
        y1[n/2] = hp_subband[n1 / 2]                              # Nyquist if N is even
    y1[n-p-t-s-1:n-p-t] = hp_subband[n1 - t - s - 1:n1 - t]             # passband (negative frequency)
    y1[n-p-t:n-p] = hp_subband[n1 - t:n1] * trans                 # transition-band (negative frequency)
    y1[n-p:n] = np.zeros(p)

    return y0 + y1
