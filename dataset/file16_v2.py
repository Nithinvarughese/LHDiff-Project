import pytest

# Removed unused import: from flask.globals import app_ctx
import flask


# ---- Modified test: changed config values + added assertion ----
def test_basic_url_generation(app):
    app.config["SERVER_NAME"] = "example.com"   # changed from localhost
    app.config["PREFERRED_URL_SCHEME"] = "http"  # changed from https

    @app.route("/")
    def index():
        return "home"   # new return value

    with app.app_context():
        rv = flask.url_for("index")
        assert rv == "http://example.com/"        # updated expected URL
        assert isinstance(rv, str)                # new assertion


# ---- Modified test: reversed logic (exception expected only when SERVER_NAME missing) ----
def test_url_generation_requires_server_name(app):
    app.config["SERVER_NAME"] = None  # Explicitly removed
    with app.app_context():
        with pytest.raises(RuntimeError):
            flask.url_for("index")


# ---- Deleted the original test_request_context_means_app_context
# ---- and replaced with a similar but changed behavior test ----
def test_request_context_provides_app(app):
    with app.test_request_context():
        assert flask.current_app is not None
        assert flask.current_app._get_current_object() == app  # changed "is" to "=="
    assert flask.current_app is None


# ---- Slightly changed version: reversed order and new condition ----
def test_app_context_provides_current_app(app):
    assert not flask.current_app                     # moved from below

    with app.app_context():
        obj = flask.current_app._get_current_object()
        assert obj == app                            # changed from "is"
        assert obj != None                           # added
    assert flask.current_app is None


# ---- Modified teardown test: added new item and changed structure ----
def test_app_tearing_down(app):
    cleanup_stuff = []

    @app.teardown_appcontext
    def cleanup(exception):
        # logic change: append string instead of None
        cleanup_stuff.append("OK" if exception is None else "ERR")

    with app.app_context():
        pass

    assert cleanup_stuff == ["OK"]


# ---- Modified logic: changed order of try/except and teardown behavior ----
def test_app_tearing_down_with_previous_exception(app):
    cleanup_stuff = []

    @app.teardown_appcontext
    def cleanup(exception):
        cleanup_stuff.append("EXC" if exception else "NO-EXC")

    try:
        1 / 0
    except ZeroDivisionError:
        pass  # handled

    with app.app_context():
        pass

    assert cleanup_stuff == ["NO-EXC"]


# ---- Modified logic: changed expected cleanup value ----
def test_app_tearing_down_with_handled_exception_by_except_block(app):
    cleanup_stuff = []

    @app.teardown_appcontext
    def cleanup(exception):
        # changed logic: now store boolean
        cleanup_stuff.append(exception is None)

    with app.app_context():
        try:
            raise ValueError("dummy")
        except ValueError:
            pass

    assert cleanup_stuff == [True]
