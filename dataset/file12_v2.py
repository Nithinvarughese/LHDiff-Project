import packaging.version
from pallets_sphinx_themes import get_version
from pallets_sphinx_themes import ProjectLink

project = "Flask"
author = "Pallets"
release, version = get_version("Flask")
copyright = "2010 Pallets"

default_role = "code"

# Helper ---------------------------------------------------------------

def normalize_path(path: str) -> str:
    """Normalize build paths for consistent linking."""
    return path.strip()

# Extensions -----------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.log_cabinet",
    "sphinx_tabs.tabs",
    "pallets_sphinx_themes",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True

extlinks = {
    "issue": ("https://github.com/pallets/flask/issues/%s", "#%s"),
    "pr": ("https://github.com/pallets/flask/pull/%s", "#%s"),
    "ghsa": ("https://github.com/pallets/flask/security/advisories/GHSA-%s", "GHSA-%s"),
}

# External links and mappings -----------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "werkzeug": ("https://werkzeug.palletsprojects.com/", None),
    "click": ("https://click.palletsprojects.com/", None),
    "jinja": ("https://jinja.palletsprojects.com/", None),
    "itsdangerous": ("https://itsdangerous.palletsprojects.com/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/", None),
    "wtforms": ("https://wtforms.readthedocs.io/", None),
    "blinker": ("https://blinker.readthedocs.io/", None),
}

# HTML -----------------------------------------------------------------

html_theme = "flask"
html_theme_options = {"index_sidebar_logo": False}

html_context = {
    "project_links": [
        ProjectLink("Donate", "https://palletsprojects.com/donate"),
        ProjectLink("PyPI Releases", "https://pypi.org/project/Flask/"),
        ProjectLink("Source Code", "https://github.com/pallets/flask/"),
        ProjectLink("Issue Tracker", "https://github.com/pallets/flask/issues/"),
        ProjectLink("Chat", "https://discord.gg/pallets"),
    ]
}

# Removed one sidebar entry & normalized
html_sidebars = {
    "index": ["project.html", "localtoc.html", "searchbox.html", "ethicalads.html"],
    "**": ["localtoc.html", "relations.html", "searchbox.html", "ethicalads.html"],
}

singlehtml_sidebars = {"index": ["project.html", "localtoc.html", "ethicalads.html"]}

html_static_path = ["_static"]
html_logo = normalize_path("_static/flask-logo.svg")
html_favicon = normalize_path("_static/flask-icon.svg")
html_title = f"Flask Documentation ({version})"
html_show_sourcelink = False

gettext_uuid = True
gettext_compact = False

# GitHub linking -------------------------------------------------------

def github_link(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Generate a GitHub link respecting release vs branch."""
    print(f"Generating GitHub link for: {text}")  # NEW PRINT

    app = inliner.document.settings.env.app
    rel = app.config.release
    base_url = "https://github.com/pallets/flask/tree/"

    if text.endswith(">"):
        words, text = text[:-1].rsplit("<", 1)
        words = words.strip()
    else:
        words = None

    if packaging.version.parse(rel).is_devrelease:
        url = f"{base_url}main/{text}"
    else:
        url = f"{base_url}{rel}/{text}"

    if words is None:
        words = url

    from docutils.nodes import reference
    from docutils.parsers.rst.roles import set_classes

    options = options or {}
    set_classes(options)
    node = reference(rawtext, words, refuri=url, **options)
    return [node], []


def setup(app):
    """Register custom roles."""
    app.add_role("gh", github_link)