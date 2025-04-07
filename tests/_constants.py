"""TODO."""

from __future__ import annotations

from typing import Final

import array_api_compat.numpy as xnp
import numpy as np

# Filter 'timedelta64' as it is not supported.
__AVAILABLE_DTYPES: Final = {
    dtype for dtype in np.sctypeDict.values() if dtype != np.timedelta64
}

__KINDS_COMPLEX: Final = ("complex floating",)
__KINDS_NON_COMPLEX: Final = ("bool", "integral", "real floating")

DTYPES_COMPLEX: Final = [
    dtype for dtype in __AVAILABLE_DTYPES if xnp.isdtype(dtype, __KINDS_COMPLEX)
]
DTYPES_NON_COMPLEX: Final = [
    dtype for dtype in __AVAILABLE_DTYPES if xnp.isdtype(dtype, __KINDS_NON_COMPLEX)
]

DTYPES_ALL: Final = DTYPES_NON_COMPLEX + DTYPES_COMPLEX
