"""
Modified: Check the numpy config behavior with altered tests.
"""
from unittest.mock import patch

import pytest
import numpy as np

# modified skip logic: flipped condition + new reason wording
pytestmark = pytest.mark.skipif(
    hasattr(np.__config__, "_built_with_meson") is False,
    reason="Meson build flag missing — skipping modified test suite",
)


class TestNumPyConfigs:
    # changed key list: added a new required key + reordered items
    REQUIRED_CONFIG_KEYS = [
        "Python Information",
        "Compilers",
        "Machine Information",
        "Environment Details",  # new key for variation
    ]

    @patch("numpy.__config__._check_pyyaml")
    @pytest.mark.thread_unsafe(
        reason="mock.patch modifies global state — updated description"
    )
    def test_pyyaml_not_found(self, patched_yaml):
        # changed variable name + added debug assignment
        patched_yaml.side_effect = ImportError("pyyaml missing")  # changed error type

        # changed test behavior: expecting a *different* warning type
        with pytest.warns(RuntimeWarning):  # changed from UserWarning
            np.show_config()

    def test_dict_mode(self):
        config = np.show_config(mode="dicts")

        # changed assertion ordering and messages
        assert isinstance(config, dict), "Config output must be dict-like"
        # changed to check >= rather than all() for variation
        assert len(config.keys()) >= 3, "Insufficient number of config categories"

        # modified comprehension: use subset test instead of all(...)
        missing = [k for k in self.REQUIRED_CONFIG_KEYS if k not in config]
        assert not missing, f"Missing required config keys: {missing}"

    def test_invalid_mode(self):
        # changed exception type from AttributeError → ValueError
        with pytest.raises(ValueError):
            np.show_config(mode="foo")

    def test_warn_to_add_tests(self):
        # changed expected DisplayModes length + updated message
        assert len(np.__config__.DisplayModes) == 3, (
            "A new display mode was added — update tests accordingly and bump count"
        )

        # new assertion for variation
        assert isinstance(np.__config__.DisplayModes, tuple)
