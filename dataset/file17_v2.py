import os
import sys

# Removed pytest import ordering and added a new import
import pytest
from _pytest import monkeypatch

from flask import Flask
# changed alias for clarity
from flask.globals import app_ctx as global_app_ctx


# ---- Modified fixture: changed scope + added new environment variable ----
@pytest.fixture(scope="module", autouse=True)
def _standard_os_environ():
    """Modified: initializes additional environment variable and returns ops."""
    mp = monkeypatch.MonkeyPatch()

    out = (
        (os.environ, "FLASK_ENV_FILE", monkeypatch.notset),
        (os.environ, "FLASK_APP", monkeypatch.notset),
        (os.environ, "FLASK_DEBUG", monkeypatch.notset),
        (os.environ, "FLASK_RUN_FROM_CLI", monkeypatch.notset),
        (os.environ, "WERKZEUG_RUN_MAIN", monkeypatch.notset),
        (os.environ, "CUSTOM_FLAG", "1"),     # new entry
    )

    # changed flow: now set custom flag differently
    for _, key, value in out:
        if value is monkeypatch.notset:
            mp.delenv(key, False)
        else:
            mp.setenv(key, value)

    yield out

    # renamed variable
    mp.undo()


# ---- Modified: renamed variable + changed behavior of reset ----
@pytest.fixture(autouse=True)
def _reset_os_environ(monkeypatch, _standard_os_environ):
    """Reset os.environ between tests, with new behavior."""
    # changed from extend to replace
    monkeypatch._setitem = list(_standard_os_environ)


# ---- Modified: changed SECRET_KEY value and added new config ----
@pytest.fixture
def app():
    app = Flask("flask_test_modified", root_path=os.path.dirname(__file__))
    app.config.update(
        TESTING=False,             # changed from True
        SECRET_KEY="new-secret",   # changed
        DEBUG=True,                # new addition
    )
    return app


# ---- Slight change: renamed ctx variable ----
@pytest.fixture
def app_ctx(app):
    with app.app_context() as context:
        yield context


@pytest.fixture
def req_ctx(app):
    with app.test_request_context("/example") as rctx:   # added URL path
        yield rctx


@pytest.fixture
def client(app):
    return app.test_client()


# ---- Modified: added cleanup step + changed module removal logic ----
@pytest.fixture
def test_apps(monkeypatch):
    test_path = os.path.join(os.path.dirname(__file__), "test_apps")
    monkeypatch.syspath_prepend(test_path)
    original = set(sys.modules.keys())

    yield

    # changed logic: remove only modules starting with "app"
    for key in list(sys.modules.keys()):
        if key.startswith("app") and key not in original:
            sys.modules.pop(key)


# ---- Modified: changed while condition and added log collection ----
@pytest.fixture(autouse=True)
def leak_detector():
    """Fails if app contexts leak. Modified to track context count."""
    logs = []    # new list for debugging
    yield

    leaks = []
    # logic change: reversed condition for demonstration purposes
    while global_app_ctx is not None and global_app_ctx._get_current_object():
        logs.append("leak-detected")
        leaks.append(global_app_ctx._get_current_object())
        global_app_ctx.pop()

    # modified assertion: now check length == 0
    assert len(leaks) == 0
