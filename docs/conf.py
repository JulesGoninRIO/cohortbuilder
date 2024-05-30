import os
import sys
import subprocess


# Add path
sys.path.insert(0, os.path.abspath('../'))

# Project information
project = 'Cohort Builder'
copyright = '2023, Fondation Asile des Aveugles'
author = 'Sepehr Mousavi and Laurent Brock'
try:
    release = subprocess.check_output(['git', 'describe']).decode().strip()
except:
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    release = 'dev' + '-' + commit
version = release

# -- General configuration -----------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_design',
    'sphinx_copybutton',
    'sphinxcontrib.autoprogram',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'm2r',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for todos  --------------------------------------------------
todo_include_todos = True
todo_link_only = True

# -- Options for HTML output ---------------------------------------------

# The theme to use for HTML and HTML Help pages.
# See the documentation for a list of builtin themes.
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'external_links': [],
    'footer_start': ['sphinx-version'],
    'gitlab_url': 'https://git.dcc.sib.swiss/cog/cohort_builder',
    'logo': {'image_dark': '_static/img/header_dark.png'},
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
}

# Add header logo and icon for HTML
html_logo = '_static/img/header.png'
html_favicon = '_static/img/icon.png'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f'Cohort Builder {release} Documentation'

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = 'Cohort Builder'

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/gettingstarted.css',
    'css/cohort_builder.css',
]

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# For referencing external packages
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'requests': ('https://requests.readthedocs.io/en/latest/', None),
    'paramiko': ('https://docs.paramiko.org/en/latest/', None),
    'lxml': ('https://lxml.de/apidoc/', None),
    'tqdm': ('https://tqdm.github.io/docs/', '_static/inv/objects-tqdm.inv'),
}

# -- Options for autodoc ------------------------------------------------------

# Set the ordering of the members of an autodoc
autodoc_member_order = 'bysource'
# Enable docstring inheritance
autodoc_inherit_docstrings = False
# Enable 'expensive' imports for sphinx_autodoc_typehints
set_type_checking_flag = True
# Show type hints in the description
autodoc_typehints = 'signature'
# Don't show the library name in type hints
autodoc_typehints_format = 'short'
# Add parameter types if the parameter is documented in the docstring
autodoc_typehints_description_target = 'documented_params'
# Enable overriding of function signatures in the first line of the docstring
autodoc_docstring_signature = True
# Make `somename` a cross-reference to a python object by default
default_role = 'py:obj'
# Add type aliases for custom type annotations
autodoc_type_aliases = {}
# Turn on sphinx.ext.autosummary
autosummary_generate = True


# Replace type hint keywords
def autodoc_process_docstring(app, what, name, obj, options, lines):
    for i in range(len(lines)):
        lines[i] = lines[i].replace('np.', 'numpy.')
        lines[i] = lines[i].replace('pd.', 'pandas.')
        lines[i] = lines[i].replace('List[', '~typing.List[')
def setup(app):
    app.connect('autodoc-process-docstring', autodoc_process_docstring)
