import os

import numpy as np
from scipy.io.wavfile import read
from scipy.signal.signaltools import resample
from scipy.signal.windows import blackman

from .other import get_package_path


def low_resonance_test_signal() -> np.ndarray:
    n2, n4, n6 = 40, 20, 80
    v2 = np.sin(0.1 * np.pi * np.arange(n2)) * blackman(n2)
    v4 = np.sin(0.2 * np.pi * np.arange(n4)) * blackman(n4)
    v6 = np.sin(0.05 * np.pi * np.arange(n6)) * blackman(n6)
    test_signal = np.zeros(512)
    test_signal[50 : n2 + 50] = v2
    test_signal[300 : n4 + 300] += v4
    test_signal[150 : n6 + 150] += v6
    return test_signal


def high_resonance_test_signal() -> np.ndarray:
    n1, n3, n5 = 160, 80, 320
    v1 = np.sin(0.1 * np.pi * np.arange(n1)) * blackman(n1)
    v3 = np.sin(0.2 * np.pi * np.arange(n3)) * blackman(n3)
    v5 = np.sin(0.05 * np.pi * np.arange(n5)) * blackman(n5)
    test_signal = np.zeros(512)
    test_signal[10 : 10 + n1] += v1
    test_signal[150 : 150 + n3] += v3
    test_signal[180 : 180 + n5] += v5
    return test_signal


def speech_signal() -> np.ndarray:
    test_signal = []
    with open(
        os.path.join(
            get_package_path().replace("/tqwt_tools/", "/"),
            "tests",
            "resources",
            "speech1.txt",
        ),
        "r",
    ) as speech_txt_handle:
        for line in speech_txt_handle.readlines():
            test_signal.append(float(line.replace("\n", "")))
    test_signal = np.array(test_signal) / np.max(np.abs(test_signal))
    return test_signal


def music_signal(start_sample: int = 0, end_sample: int = -1) -> np.ndarray:
    """fs = 8820Hz"""
    _, audio = read(
        os.path.join(
            get_package_path().replace("/tqwt_tools/", "/"),
            "tests",
            "resources",
            "testaudio.wav",
        )
    )
    audio = audio / np.abs(np.max(audio))
    return resample(audio, int(audio.shape[0] / 5))[start_sample:end_sample]
