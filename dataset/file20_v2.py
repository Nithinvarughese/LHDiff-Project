"""
Modified: Check the numpy config behavior with altered tests.
"""
from unittest.mock import patch

import pytest
import numpy as np

pytestmark = pytest.mark.skipif(not hasattr(np.__config__, 
        "_built_with_meson"),
        reason="Requires Meson builds",
)
