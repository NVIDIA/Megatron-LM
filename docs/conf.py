# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Megatron-LM"
copyright = "2025, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "latest"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For our markdown docs
    "sphinx.ext.viewcode",  # For adding a link to view source code in docs
    "sphinx.ext.doctest",  # Allows testing in docstrings
    "sphinx.ext.napoleon",  # For google style docstrings
    "sphinx_copybutton",  # For copy button in code blocks
]

# Check if we should skip autodoc generation
# usage: SKIP_AUTODOC=true
skip_autodoc = os.environ.get("SKIP_AUTODOC", "false").lower() == "true"

if not skip_autodoc:
    extensions.append("autodoc2")  # Generates API docs

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for MyST Parser (Markdown) --------------------------------------
# MyST Parser settings
myst_enable_extensions = [
    "dollarmath",  # Enables dollar math for inline math
    "amsmath",  # Enables LaTeX math for display mode
    "colon_fence",  # Enables code blocks using ::: delimiters instead of ```
    "deflist",  # Supports definition lists with term: definition format
    "fieldlist",  # Enables field lists for metadata like :author: Name
    "tasklist",  # Adds support for GitHub-style task lists with [ ] and [x]
    "attrs_block",  # Enables setting attributes on block elements using {#id .class key=val}
]
myst_heading_anchors = 5  # Generates anchor links for headings up to level 5

# -- Options for Autodoc2 ---------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

if not skip_autodoc:
    autodoc2_packages = [
        {
            "path": "../megatron/core",  # Path to your package relative to conf.py
            "exclude_dirs": ["converters"],  # list of directory names to exclude
        }
    ]
    autodoc2_render_plugin = "myst"  # Use MyST for rendering docstrings
    autodoc2_output_dir = "apidocs"  # Output directory for autodoc2 (relative to docs/)
    # This is a workaround that uses the parser located in autodoc2_docstrings_parser.py to allow autodoc2 to
    # render google style docstrings.
    # Related Issue: https://github.com/sphinx-extensions2/sphinx-autodoc2/issues/33
    autodoc2_docstring_parser_regexes = [
        (r".*", "docs.autodoc2_docstrings_parser"),
    ]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "switcher": {
        "json_url": "versions1.json",
        "version_match": release,
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/Megatron-LM/",
            "icon": "fa-brands fa-github",
        }
    ],
    "extra_head": {
        """
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    """
    },
    "extra_footer": {
        """
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    """
    },
}
html_extra_path = ["project.json", "versions1.json"]

# Github links are now getting rate limited from the Github Actions
linkcheck_ignore = [
    ".*github\\.com.*",
    ".*githubusercontent\\.com.*",
]
