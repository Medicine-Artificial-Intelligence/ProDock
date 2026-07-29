import os
import sys
from importlib.metadata import PackageNotFoundError, version as _get_version

sys.path.insert(0, os.path.abspath(".."))

project = "ProDock"
author = "Tieu-Long Phan"

try:
    release = _get_version("prodock")
except PackageNotFoundError:
    release = "0.5.0"

version = ".".join(release.split(".")[:3])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = False
add_module_names = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_class_signature = "mixed"
maximum_signature_line_length = 88
python_use_unqualified_type_names = False
nitpicky = False

# Heavy scientific/runtime deps are mocked so the docs build without the full stack.
autodoc_mock_imports = [
    "rdkit",
    "rdkit.Chem",
    "rdkit.Chem.rdchem",
    "pandas",
    "numpy",
    "yaml",
    "vina",
    "pymol",
    "openbabel",
    "meeko",
    "pdbfixer",
    "openmm",
    "Bio",
    "Bio.PDB",
    "MDAnalysis",
    "prolif",
    "matplotlib",
    "matplotlib.pyplot",
    "joblib",
    "IPython",
    "ipywidgets",
    "py3Dmol",
    "tqdm",
    "interactions",
]

suppress_warnings = [
    "ref.python",
    "autosectionlabel.*",
    "autodoc.import_object",
]

html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_logo = "logo-light.svg"
html_favicon = "favicon.svg"
html_show_sourcelink = False
html_copy_source = False
html_show_sphinx = False
html_last_updated_fmt = "%Y-%m-%d"
html_context = {"default_mode": "light"}

try:
    import pydata_sphinx_theme  # noqa: F401

    html_theme = "pydata_sphinx_theme"
except Exception:
    html_theme = "alabaster"

if html_theme == "pydata_sphinx_theme":
    # Left sidebar = section navigation only
    html_sidebars = {
        "**": ["sidebar-nav-bs"],
    }

    html_theme_options = {
        "logo": {
            "image_light": "logo-light.svg",
            "image_dark": "logo-dark.svg",
            "text": "",
            "alt_text": "ProDock documentation",
        },
        "show_prev_next": True,
        "navigation_depth": 2,
        "show_nav_level": 2,
        "show_toc_level": 2,
        "navbar_align": "left",
        "header_links_before_dropdown": 8,
        "navbar_start": ["navbar-logo"],
        "navbar_center": ["navbar-nav"],
        "navbar_end": ["theme-switcher", "navbar-icon-links", "search-button"],
        # Right sidebar = current page TOC
        "secondary_sidebar_items": ["page-toc"],
        "announcement": (
            "Multi-receptor, multi-ligand, multi-engine molecular docking "
            "with reusable storage."
        ),
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/Medicine-Artificial-Intelligence/prodock",
                "icon": "fa-brands fa-github",
                "type": "fontawesome",
            },
            {
                "name": "Issues",
                "url": "https://github.com/Medicine-Artificial-Intelligence/prodock/issues",
                "icon": "fa-solid fa-circle-question",
                "type": "fontawesome",
            },
            {
                "name": "Releases",
                "url": "https://github.com/Medicine-Artificial-Intelligence/prodock/releases",
                "icon": "fa-solid fa-tag",
                "type": "fontawesome",
            },
            {
                "name": "PyPI",
                "url": "https://pypi.org/project/prodock/",
                "icon": "fa-solid fa-cube",
                "type": "fontawesome",
            },
        ],
    }
else:
    html_theme_options = {}
