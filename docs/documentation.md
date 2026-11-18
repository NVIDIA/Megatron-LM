---
orphan: true
---

{/*
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
*/}


# Build the Documentation

The published Megatron Core documentation is a [Fern](https://buildwithfern.com/learn/docs/getting-started/quickstart)
site. Fern compiles Markdown/MDX content and YAML configuration into a hosted static site.
The content lives in `docs/`, and the Fern configuration lives in `fern/`.

    - [Prerequisites](#prerequisites)
    - [Install the Fern CLI](#install-the-fern-cli)
    - [Project Layout](#project-layout)
    - [Add or Edit a Page](#add-or-edit-a-page)
    - [Validate the Documentation](#validate-the-documentation)
    - [Preview Locally](#preview-locally)
    - [Share a Preview](#share-a-preview)
    - [Publish](#publish)

### Prerequisites

* Node.js 22
* The Fern CLI at the version pinned in `fern/fern.config.json`. Using the pinned version keeps
  local results consistent with CI.

### Install the Fern CLI

From the repository root, install the pinned version of the CLI:

```sh
npm install -g "fern-api@$(jq -er .version fern/fern.config.json)"
```

> Do not run `fern init` in this repository. The `fern/` folder already exists and is
> configured. Run `fern upgrade` only when intentionally bumping the CLI pin in
> `fern/fern.config.json`.

### Project Layout

| Path | Purpose |
| --- | --- |
| `fern/fern.config.json` | Identifies the Fern organization and pins the CLI version. |
| `fern/docs.yml` | Site manifest: docs URL, title, theme, navbar links, redirects, and versions. |
| `fern/redirects.yml` | Redirects from the legacy URLs to the Fern site. |
| `fern/versions/` | The **Archives** version, which links to previous developer guides. |
| `docs/index.yml` | Navigation for the **Nightly** version (sidebar structure and page slugs). |
| `docs/**/*.md` | The documentation pages. |

The site has two versions: **Nightly** (the default, built from `docs/`) and **Archives**.
Publishing tags select a Git revision. 

### Add or Edit a Page

1. Create or edit a Markdown file under `docs/`. Start a new page with front matter:

   ```markdown
   ---
   title: "Page Title"
   description: "Optional subtitle"
   ---

   Hello world!
   ```

2. Register the page in `docs/index.yml` so it appears in the navigation. Paths are relative to
   `docs/index.yml`:

   ```yaml
   navigation:
     - section: "Basic Usage"
       skip-slug: true
       contents:
         - page: "Hello World"
           path: ./user-guide/hello-world.md
           slug: hello-world
   ```

Fern supports GitHub flavored Markdown within MDX files. Pages are parsed as MDX, so escape or put
in code spans any bare `<`, `>`, or `{` characters in prose.
If you rename or move a page, add a rule to `fern/redirects.yml` for the old URL.

### Validate the Documentation

Run the same checks as CI from the `fern/` folder:

```sh
cd fern/
fern check
fern docs md check
fern docs broken-links --strict
```

* `fern check` validates the Fern configuration.
* `fern docs md check` validates the MDX content.
* `fern docs broken-links --strict` fails on broken internal links.

### Preview Locally

Run the local development server with hot reloading. The site updates as you edit Markdown and
YAML files:

```sh
cd fern/
fern docs dev
```

Open the URL printed in the terminal (by default, `http://localhost:3000`) to view the site.
If you use a remote machine, forward the port to your local machine.

`fern docs dev` cannot test redirects. Verify them on a shared preview deployment.

### Share a Preview

You do not need to create previews manually for pull requests. Once a non-draft PR is synchronized to
`pull-request/<number>`, CI validates the documentation and, after a reviewer approves the
`fern-preview` deployment, posts a preview link on the PR. See `fern/README.md` for details.

To generate a shareable preview link manually, log in with `fern login` (or set `FERN_TOKEN`) and run:

```sh
cd fern/
fern generate --docs --preview
```

### Publish

Publishing is handled by CI and requires approval. Do not publish from a local machine.

* Pushes to `main` that change docs update the shared `megatron-lm-staging` preview.
* Pushing a `docs/v*` tag, or manually running the `fern-docs-publish.yml` workflow, publishes
  production after validation and approval of the `fern-publish` environment.
* When a final `core_vX.Y.Z` release is cut, a draft PR adds the release to `fern/versions/archives.mdx`.



