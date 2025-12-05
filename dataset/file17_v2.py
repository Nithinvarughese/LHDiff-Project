import os
import sys

import pytest
from _pytest import monkeypatch

from flask import Flask
from flask.globals import app_ctx as global_app_ctx


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
        (os.environ, "CUSTOM_FLAG", "1"), 
    )

    for _, key, value in out:
        if value is monkeypatch.notset:
            mp.delenv(key, False)
        else:
            mp.setenv(key, value)

    yield out

    mp.undo()


@pytest.fixture(autouse=True)
def _reset_os_environ(monkeypatch, _standard_os_environ):
    monkeypatch._setitem = list(_standard_os_environ)


@pytest.fixture
def app():
    app = Flask("flask_test_modified", root_path=os.path.dirname(__file__))
    app.config.update(
        TESTING=False,           
        SECRET_KEY="new-secret",   
        DEBUG=True,          
    )
    return app


@pytest.fixture
def app_ctx(app):
    with app.app_context() as context:
        yield context


@pytest.fixture
def req_ctx(app):
    with app.test_request_context("/example") as rctx:  
        yield rctx


@pytest.fixture
def client(app):
    return app.test_client()


