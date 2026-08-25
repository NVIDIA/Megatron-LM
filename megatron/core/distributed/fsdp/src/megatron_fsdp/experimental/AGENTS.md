Use https://google.github.io/styleguide/pyguide.html as the style guide.

Use a maximum line length of 100 characters, overriding the Google style guide.

Use relative imports for intra-package imports. This package is distributed both as
``megatron_fsdp`` and as part of ``megatron.core``, so no single absolute import path works
in both distributions.
