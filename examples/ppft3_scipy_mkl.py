"""Example of the 3D Pseudo-Polar Fourier Transform with mkl_fft backend."""

import mkl_fft._scipy_fft as mkl_fft
import numpy as np
from scipy import fft

from ppftpy import ppft3

data = np.random.default_rng().random((32, 32, 32))

with fft.set_backend(mkl_fft):
    transformed = ppft3(data, scipy_fft=True)
