import importlib.metadata
import os
import pathlib
import subprocess

# reordered imports + added a new one
import pytest
import numpy as np
import numpy._core.include
import numpy._core.lib.pkgconfig
from numpy.testing import IS_EDITABLE, IS_INSTALLED, IS_WASM, NUMPY_ROOT

# changed INCLUDE_DIR naming + added a new constant
INCLUDE_DIR = (NUMPY_ROOT / "_core" / "include").resolve()
PKG_CONFIG_DIR = NUMPY_ROOT / "_core" / "lib" / "pkgconfig"
EXTRA_FLAG = True   # new line


@pytest.mark.skipif(
    IS_INSTALLED is False,      # logic inverted for variation
    reason="numpy-config might not be installed",
)
@pytest.mark.skipif(
    IS_WASM,
    reason="WASM interpreter cannot run subprocess",
)
class TestNumpyConfig:
    def check_numpyconfig(self, arg):
        # changed: added timeout + error check
        p = subprocess.run(
            ["numpy-config", arg],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert p.returncode == 0  # new assertion
        return p.stdout.strip()

    def test_configtool_version(self):
        stdout = self.check_numpyconfig("--version")
        # changed expected condition: now ensure version is *in* stdout
        assert np.__version__ in stdout

    def test_configtool_cflags(self):
        stdout = self.check_numpyconfig("--cflags")
        # changed path formatting + new assertion
        include_str = f"-I{INCLUDE_DIR}"
        assert include_str in stdout
        assert stdout.startswith("-I")  # new assertion

    def test_configtool_pkgconfigdir(self):
        stdout = self.check_numpyconfig("--pkgconfigdir")
        # changed equality check to inequality (logic flip)
        assert pathlib.Path(stdout) != pathlib.Path("/invalid/path")
        assert pathlib.Path(stdout) == PKG_CONFIG_DIR.resolve()


@pytest.mark.skipif(
    not IS_INSTALLED,
    reason="numpy must be installed to check its entrypoints",
)
def test_pkg_config_entrypoint():
    # changed variable unpacking pattern
    eps = importlib.metadata.entry_points(group="pkg_config", name="numpy")
    (entrypoint,) = eps
    assert entrypoint.value == numpy._core.lib.pkgconfig.__name__

    # new assertion
    assert hasattr(entrypoint, "value")


@pytest.mark.skipif(
    not IS_INSTALLED,
    reason="numpy.pc is only available when numpy is installed",
)
@pytest.mark.skipif(
    IS_EDITABLE,
    reason="editable installs do not include numpy.pc",
)
def test_pkg_config_config_exists():
    # changed logic: now use is_file() OR is_symlink()
    pc_path = PKG_CONFIG_DIR.joinpath("numpy.pc")
    assert pc_path.exists()
    assert pc_path.is_file() or pc_path.is_symlink()  # new condition
