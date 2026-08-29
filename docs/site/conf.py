project = "Moto"
author = "Moto contributors"
root_doc = "index"

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.intersphinx",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = []
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

html_theme = "furo"
html_title = "Moto Documentation"
html_baseurl = "https://haizhouz.github.io/moto/"
html_theme_options = {
    "source_repository": "https://github.com/HaizhouZ/moto/",
    "source_branch": "main",
    "source_directory": "docs/site/",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

autoapi_dirs = ["../stubs/moto"]
autoapi_root = "reference/api"
autoapi_file_patterns = ["*.pyi"]
autoapi_ignore = ["*moto_pywrap.pyi", "*definition*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-module-summary",
    "imported-members",
]
autoapi_own_page_level = "class"
autoapi_add_toctree_entry = False
autoapi_python_use_implicit_namespaces = True
autoapi_keep_files = False
suppress_warnings = ["autoapi.python_import_resolution"]
