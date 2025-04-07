"""Example of the 2D Pseudo-Polar Fourier Transform with pyFFTW backend."""

import mkl_fft._scipy_fft as mkl_fft
import numpy as np
from scipy import fft

from ppftpy import ppft2

data = np.random.default_rng().random((128, 128))

with fft.set_backend(mkl_fft):
    transformed = ppft2(data, scipy_fft=True)
