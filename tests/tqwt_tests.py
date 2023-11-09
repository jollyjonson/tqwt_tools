import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from scipy.io.matlab import loadmat
from scipy.io.wavfile import read, write
from scipy.signal import resample
from time import time

from tqwt_tools.tqwt import synthesis_filter_bank, analysis_filter_bank
from tqwt_tools import tqwt, itqwt
from tqwt_tools.util import get_package_path
from tqwt_tools.util.test_signal import speech_signal

# these output values of the afb and sfb were generated from the Matlab toolbox ([V0, V1] = afb((0:19), 16, 12))
EXPECTEDV0 = np.array([0, 1, 2, 3, 4, 4.85268414962686, 4.24264068711929, 1.68666974424811, 0, 3.13238666788935,
                       9.89949493661167, 14.5580524488806, 16, 17, 18, 19])
EXPECTEDV1 = np.array([0, 1.20476410303436, 4.24264068711929, 6.79375780947761, 8, 9, 10, 11, 12,
                       12.6169787890298, 9.89949493661167, 3.61429230910309])
EXPECTEDY = np.array([0, 1, 2, 3, 4, 5, 9.283601844998817, 17.507536716885582, 23.775413799563122, 25.131037888449338,
                      25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37.296444516289142, 37.790891865207485,
                      31.014025662207025, 20.061288372791118, 15, 16, 17, 18, 19], dtype=complex)


class TestTQWT(unittest.TestCase):

    def test_afb(self):
        """Check the analysis filter bank function in a simple case against hard-coded Matlab results"""
        v0, v1 = analysis_filter_bank(np.arange(20).astype(float), 16, 12)
        npt.assert_allclose(EXPECTEDV0, v0)
        npt.assert_allclose(EXPECTEDV1, v1)

    def test_tqwt(self):
        """Test the tqwt main function with random numbers"""
        sine = np.cos(2 * np.pi * 2 / 10 * np.arange(48))
        w = tqwt(sine, 3, 3, 3)
        w_matlab = loadmat(os.path.join(get_package_path().replace('/tqwt_tools/', '/'),
                                        'tests', 'resources', 'w_sine'))['w']
        for subband, subband_matlab in zip(w, np.squeeze(w_matlab)):
            npt.assert_allclose(np.imag(subband), 0, atol=1e-14)
            npt.assert_allclose(np.real_if_close(subband).astype(np.float32), np.squeeze(subband_matlab))

    def test_tqwt_with_audio(self) -> None:
        """Test the TQWT with audio material"""

        #sine = np.cos(2 * np.pi * 1000 / 16000 * np.arange(48000))
        #wAudio = tqwt.tqwt(sine, 8, 2, 32)

        #start = time()
        #wAudio = tqwt.tqwt(self.audio, 10, 1.5, 67)
        #print("Runtime: ", time()-start)

        # # plot the resulting spectrogram by resampling each subband
        # spectrogram = np.zeros((wAudio.shape[0], self.audio.shape[0]), dtype=np.complex)
        # for idx, subband in enumerate(wAudio):
        #     spectrogram[idx, :] = resample(subband, self.audio.shape[0])
        # plt.imshow(20*np.log10(1e-8+np.abs(spectrogram)), aspect='auto')
        # plt.yticks([]); plt.xticks([])
        # plt.show()


class TestITQWT(unittest.TestCase):

    def test_sfb(self) -> None:
        """Test the synthesis filter bank used in the itqwt against results from the original Matlab implementation"""
        y = synthesis_filter_bank(np.arange(20), np.arange(start=20, stop=40), 30)
        npt.assert_allclose(EXPECTEDY, y)


class TestPerfectReconstruction(unittest.TestCase):

    def test_perfect_reconstruction(self) -> None:
        q = 2; r = 2; j = 8
        x = speech_signal()
        n = speech_signal().shape[0]
        w = tqwt(x, q, r, j)                      # wavelet transform
        y = itqwt(w, q, r, n)                     # inverse wavelet transform
        max_reconstruction_error_db = 20 * np.log10(max(abs(x - y)))
        self.assertLess(max_reconstruction_error_db,  -290)
        print("[TestPerfectReconstruction] Reconstrucion Error: ", max_reconstruction_error_db, "dB")


if __name__ == '__main__':
    unittest.main()
