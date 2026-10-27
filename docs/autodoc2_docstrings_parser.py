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

from docutils import nodes
from myst_parser.parsers.sphinx_ import MystParser
from sphinx.ext.napoleon.docstring import GoogleDocstring


class NapoleonParser(MystParser):
    """Add support for Google style docstrings."""

    def parse(self, input_string: str, document: nodes.document) -> None:
        """Parse Google style docstrings."""

        # Get the Sphinx configuration
        config = document.settings.env.config

        # Process with Google style
        google_parsed = str(GoogleDocstring(input_string, config))

        return super().parse(google_parsed, document)


Parser = NapoleonParser
