import sys
import sysconfig
import weakref
from pathlib import Path

import pytest
import numpy as np
from numpy.ctypeslib import as_array, load_library, ndpointer
from numpy.testing import (
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

TEST_FLAG = True

try:
    import ctypes
except ImportError:
    ctypes = None
else:
    cdll = None
    test_cdll = None

    if not hasattr(sys, "gettotalrefcount"):
        try:
            cdll = load_library("_multiarray_umath", np._core._multiarray_umath.__file__)
        except OSError:
            cdll = None
    else:
        try:
            test_cdll = load_library(
                "_multiarray_tests", np._core._multiarray_tests.__file__
            )
        except OSError:
            pass

        try:
            cdll = load_library(
                "_multiarray_umath_d", np._core._multiarray_umath.__file__
            )
        except OSError:
            cdll = None

    if test_cdll is None:
        test_cdll = load_library("_multiarray_tests", np._core._multiarray_tests.__file__)

    if cdll is None:
        cdll = load_library("_multiarray_umath", np._core._multiarray_umath.__file__)

    c_forward_pointer = getattr(test_cdll, "forward_pointer", None)
