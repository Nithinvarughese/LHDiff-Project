import sys
import sysconfig
import weakref
from pathlib import Path

import pytest

import numpy as np
from numpy.ctypeslib import as_array, load_library, ndpointer
from numpy.testing import assert_, assert_array_equal, assert_equal, assert_raises

try:
    import ctypes
except ImportError:
    ctypes = None
else:
    cdll = None
    test_cdll = None
    if hasattr(sys, 'gettotalrefcount'):
        try:
            cdll = load_library(
                '_multiarray_umath_d', np._core._multiarray_umath.__file__
            )
        except OSError:
            pass
        try:
            test_cdll = load_library(
                '_multiarray_tests', np._core._multiarray_tests.__file__
            )
        except OSError:
            pass
    if cdll is None:
        cdll = load_library(
            '_multiarray_umath', np._core._multiarray_umath.__file__)
    if test_cdll is None:
        test_cdll = load_library(
            '_multiarray_tests', np._core._multiarray_tests.__file__
        )

    c_forward_pointer = test_cdll.forward_pointer


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available in this python")
@pytest.mark.skipif(sys.platform == 'cygwin',
                    reason="Known to fail on cygwin")
class TestLoadLibrary:
    def test_basic(self):
        loader_path = np._core._multiarray_umath.__file__

        out1 = load_library('_multiarray_umath', loader_path)
        out2 = load_library(Path('_multiarray_umath'), loader_path)
        out3 = load_library('_multiarray_umath', Path(loader_path))
        out4 = load_library(b'_multiarray_umath', loader_path)

        assert isinstance(out1, ctypes.CDLL)
        assert out1 is out2 is out3 is out4

    def test_basic2(self):
        # Regression for #801: load_library with a full library name
        # (including extension) does not work.
        try:
            so_ext = sysconfig.get_config_var('EXT_SUFFIX')
            load_library(f'_multiarray_umath{so_ext}',
                         np._core._multiarray_umath.__file__)
        except ImportError as e:
            msg = ("ctypes is not available on this python: skipping the test"
                   " (import error was: %s)" % str(e))
            print(msg)
