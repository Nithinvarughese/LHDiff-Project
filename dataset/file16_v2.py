import pytest

import flask


def test_basic_url_generation(app):
    app.config["SERVER_NAME"] = "localhost"
    app.config["PREFERRED_URL_SCHEME"] = "https"

    @app.route("/")
    def index():
        return "home"

    with app.app_context():
        rv = flask.url_for("index")
        assert rv == "https://localhost/" 
        assert isinstance(rv, str)   


def test_url_generation_requires_server_name(app):
    app.config["SERVER_NAME"] = None 
    with app.app_context():
        with pytest.raises(RuntimeError):
            flask.url_for("index")


def test_request_context_provides_app(app):
    with app.test_request_context():
        assert flask.current_app is not None
        assert flask.current_app._get_current_object() == app
    assert flask.current_app is None


def test_app_context_provides_current_app(app):
    with app.app_context():
        assert flask.current_app._get_current_object() is app
    assert not flask.current_app

