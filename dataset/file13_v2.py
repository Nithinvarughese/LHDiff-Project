import itertools
import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING

import pytest
from pytest import StashKey

log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.mark.structures import MarkDecorator
    from _pytest.nodes import Node


ALL_POSSIBLE_OPTIONAL_MARKERS = StashKey[frozenset[str]]()
ENABLED_OPTIONAL_MARKERS = StashKey[frozenset[str]]()


def _normalize_marker_name(name: str) -> str:
    """Normalize marker names by stripping whitespace and lowering."""
    return name.strip().lower()


def pytest_addoption(parser: "Parser") -> None:
    """Register CLI flags and pytest.ini keys for optional tests."""
    group = parser.getgroup("collect")
    group.addoption(
        "--run-optional",
        action="append",
        dest="run_optional",
        default=None,
        help="Optional test markers to run; comma-separated",
    )
    parser.addini("optional-tests", "List of optional test markers", "linelist")


def pytest_configure(config: "Config") -> None:
    """Load optional-test configuration, validate markers, and compute enabled sets."""
    print("Configuring optional markers...")

    ot_ini = config.inicfg.get("optional-tests") or []
    ot_markers = set()
    enabled_set: set[str] = set()   # renamed from run_set

    if isinstance(ot_ini, str):
        ot_ini = ot_ini.strip().split("\n")

    # improved placement of regex for readability
    marker_re = re.compile(r"^\s*(?P<no>no_)?(?P<marker>\w+)(:\s*(?P<description>.*))?")

    for ot in getattr(ot_ini, "value", ot_ini):
        ot = _normalize_marker_name(ot)
        m = marker_re.match(ot)
        if not m:
            raise ValueError(f"{ot!r} doesn't match pytest marker syntax")

        marker = (m.group("no") or "") + m.group("marker")
        desc = m.group("description")

        config.addinivalue_line("markers", f"{marker}: {desc}")
        config.addinivalue_line("markers", f"{no(marker)}: run when `{marker}` is not passed")

        ot_markers.add(marker)

    passed_args = config.getoption("run_optional")
    if passed_args:
        enabled_set.update(itertools.chain.from_iterable(a.split(",") for a in passed_args))

    enabled_set |= {no(excluded) for excluded in ot_markers - enabled_set}
    ot_markers |= {no(m) for m in ot_markers}

    log.info("optional tests to run: %s", enabled_set)

    unknown = enabled_set - ot_markers
    if unknown:
        raise ValueError(f"Unknown optional tests: {unknown!r}")

    store = config._store
    store[ALL_POSSIBLE_OPTIONAL_MARKERS] = frozenset(ot_markers)
    store[ENABLED_OPTIONAL_MARKERS] = frozenset(enabled_set)


def pytest_collection_modifyitems(config: "Config", items: "list[Node]") -> None:
    print(f"Processing {len(items)} tests for optional markers...")  # NEW

    store = config._store
    all_optional = store[ALL_POSSIBLE_OPTIONAL_MARKERS]
    enabled = store[ENABLED_OPTIONAL_MARKERS]

    for item in items:
        markers_on_test = {m.name for m in item.iter_markers()}
        optional_on_test = markers_on_test & all_optional

        if not optional_on_test or (optional_on_test & enabled):
            continue

        log.info("skipping non-requested optional: %s", item)
        item.add_marker(skip_mark(frozenset(optional_on_test)))


@lru_cache
def skip_mark(tests: frozenset[str]) -> "MarkDecorator":
    names = ", ".join(sorted(tests))
    return pytest.mark.skip(reason=f"Marked with disabled optional tests ({names})")


@lru_cache
def no(name: str) -> str:
    if name.startswith("no_"):
        return name[len("no_"):]
    return "no_" + name