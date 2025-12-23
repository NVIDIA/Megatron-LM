---
orphan: true
---

# Documentation Development

- [Documentation Development](#documentation-development)
  - [Build the Documentation](#build-the-documentation)
  - [Live Building](#live-building)
  - [Documentation Version](#documentation-version)


## Build the Documentation

The following sections describe how to set up and build the NeMo RL documentation.

Switch to the documentation source folder and generate HTML output.

```sh
cd docs/
uv run --group docs sphinx-build . _build/html
```

* The resulting HTML files are generated in a `_build/html` folder that is created under the project `docs/` folder.
* The generated python API docs are placed in `apidocs` under the `docs/` folder.

## Checking for Broken Links

To check for broken http links in the docs, run this command:

```sh
cd docs/
uv run --group docs sphinx-build --builder linkcheck . _build/linkcheck
```

It will output a JSON file at `_build/linkcheck/output.json` with links it found while building the
docs. Records will have a status of `broken` if the link is not reachable. The `docs/conf.py` file is
configured to ignore github links because the CI test will often experience rate limit errors.
Comment out the `linkcheck_ignore` variable there to check all the links.

## Live Building

When writing documentation, it can be helpful to serve the documentation and have it update live while you edit.

To do so, run:

```sh
cd docs/
uv run --group docs sphinx-autobuild . _build/html --port 12345 --host 0.0.0.0
```

Open a web browser and go to `http://${HOST_WHERE_SPHINX_COMMAND_RUN}:12345` to view the output.

## Documentation Version

The three files below control the version switcher. Before you attempt to publish a new version of the documentation, update these files to match the latest version numbers.

* docs/versions1.json
* docs/project.json
* docs/conf.py
