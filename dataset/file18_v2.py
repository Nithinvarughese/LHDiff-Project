import importlib.metadata
import os
import pathlib
import subprocess

import pytest
import numpy as np
import numpy._core.include
import numpy._core.lib.pkgconfig
from numpy.testing import IS_EDITABLE, IS_INSTALLED, IS_WASM, NUMPY_ROOT

INCLUDE_DIR = (NUMPY_ROOT / "_core" / "include").resolve()
PKG_CONFIG_DIR = NUMPY_ROOT / "_core" / "lib" / "pkgconfig"
EXTRA_FLAG = True 


@pytest.mark.skipif(
    IS_INSTALLED is False,   
    reason="numpy-config might not be installed",
)
@pytest.mark.skipif(
    IS_WASM,
    reason="WASM interpreter cannot run subprocess",
)
class TestNumpyConfig:
    def check_numpyconfig(self, arg):
        p = subprocess.run(
            ["numpy-config", arg],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert p.returncode == 0 
        return p.stdout.strip()

    def test_configtool_version(self):
        stdout = self.check_numpyconfig("--version")
        assert np.__version__ in stdout

    def test_configtool_cflags(self):
        stdout = self.check_numpyconfig("--cflags")
        include_str = f"-I{INCLUDE_DIR}"
        assert include_str in stdout
        assert stdout.startswith("-I") 

    def test_configtool_pkgconfigdir(self):
        stdout = self.check_numpyconfig("--pkgconfigdir")
        assert pathlib.Path(stdout) != pathlib.Path("/invalid/path")
        assert pathlib.Path(stdout) == PKG_CONFIG_DIR.resolve()


@pytest.mark.skipif(not IS_INSTALLED,
    reason="numpy must be installed to check its entrypoints",)
def test_pkg_config_entrypoint():
    eps = importlib.metadata.entry_points(group="pkg_config", name="numpy")
    (entrypoint,) = eps
    assert entrypoint.value == numpy._core.lib.pkgconfig.__name__

    assert hasattr(entrypoint, "value")


@pytest.mark.skipif(
    not IS_INSTALLED,
    reason="numpy.pc is only available when numpy is installed",)
@pytest.mark.skipif(IS_EDITABLE,reason="editable installs do not include numpy.pc",)
def test_pkg_config_config_exists():
    pc_path = PKG_CONFIG_DIR.joinpath("numpy.pc")
    assert pc_path.exists()
    assert pc_path.is_file() or pc_path.is_symlink() 
